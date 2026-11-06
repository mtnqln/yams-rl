from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from yams_rl.rl.config import Args
from yams_rl.rl.model import Agent
from yams_rl.rl.vec_env import SlowVecEnv


class Logger(Protocol):
    def add_scalar(self,tag,scalar_value,global_step)->None:
        ...


@dataclass
class HeavyArgs:
    seed: np.random.Generator
    envs: SlowVecEnv
    agent:Agent
    optimizer: torch.optim.AdamW
    logger: Logger
    logger_step: int

    next_mask: torch.Tensor
    next_obs: torch.Tensor
    next_done: torch.Tensor

def run_one_ppo_iter(args:Args,heavy_args:HeavyArgs):
    """
    Execute one iteration of PPO algorithm : \\theta_{k+1} = argmax_{\\theta}E_{s,a~\\theta_k}[L(s,a,\\theta_k,\\theta)]
    """
    seed = heavy_args.seed.integers(low=1,high=2**30)
    optimizer = heavy_args.optimizer

    slow_vec_envs = heavy_args.envs
    agent = heavy_args.agent

    # Buffers; it can be done upstream in case of GPU training or very big buffer
    # ty is not very happy but there is broadcasting : (a,b) + (c,) = (a,b,c)
    observations = torch.zeros(size=(args.num_step,args.num_envs)+slow_vec_envs.single_observation_space.shape)  # ty: ignore[unsupported-operator]
    values = torch.zeros(size=(args.num_step,args.num_envs)) # each value is also a scalar for one state
    log_pol = torch.zeros(size=(args.num_step,args.num_envs)) # each logpol is a scalar because it is taken for one state and action
    rewards = torch.zeros(size=(args.num_step,args.num_envs)) # on reward at each step
    actions = torch.zeros(size=(args.num_step,args.num_envs))
    done = torch.zeros(size=(args.num_step,args.num_envs))
    masks = torch.zeros(size=(args.num_step,args.num_envs,slow_vec_envs.single_action_space.n))
    scores = torch.zeros_like(done)
    entropys = torch.zeros_like(log_pol)

    next_obs = heavy_args.next_obs
    next_mask = heavy_args.next_mask
    done[0] = heavy_args.next_done

    ### Collecting the env / Rollout ###
    for step in range(args.num_step):

        with torch.no_grad():
            action,log_policy,entropy,value = agent.get_action_and_value(next_obs,next_mask)
            obs, reward, terminated, truncated, info = slow_vec_envs.step(action)

            mask = slow_vec_envs.action_masks()

            log_pol[step] = log_policy
            entropys[step] = entropy
            values[step] = value
            rewards[step] = reward
            done[step] = torch.logical_or(terminated,truncated)
            scores[step] = info["agents_scores"]
            actions[step]=action
            observations[step] = next_obs # We save obs_t before changing it to obs_{t+1} to match with action_t
            masks[step] = next_mask
            next_obs = obs
            next_mask = mask

    ### GAE ###
     # this is V(s_{t+1}) because the tensors stop at s_t and V(s_t) we use the last next_obs which is s_t+1
     # done[t]=1 means this is the last round of the game and t+1 will be the start of a new game
     # when we are at the last round of the game we force V(s_{t+1}) to be 0 (with (1-done[t])) since there is no advantage after it
    with torch.no_grad():
        bootstrap_value = agent.get_value(next_obs).flatten()
        advantage_GAE = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(args.num_step)):
            if t == args.num_step - 1:
                next_values = bootstrap_value
            else:
                next_values = values[t+1]
            delta_t = rewards[t] + args.gae_gamma*next_values*(1-done[t]) - values[t]
            advantage_GAE[t] = delta_t + args.gae_gamma*args.gae_lambda*last_gae*(1-done[t])
            last_gae = advantage_GAE[t]

    ### Reward to go ###
    with torch.no_grad():
        reward_to_go = torch.zeros_like(rewards)
        last_reward_to_go = args.gae_gamma*bootstrap_value # we multiply by gamma to keep the same "scale" with V(s_T) in GAE
        for t in reversed(range(args.num_step)):
            reward_to_go[t] = rewards[t] + last_reward_to_go*(1-done[t])
            last_reward_to_go = reward_to_go[t]

    agent.train()
    _,log_policy,entropy,value = agent.get_action_and_value(observations,actions,masks)
    new_log_pol = log_policy
    values = agent.get_value(observations)

    ### Optimizing ###
    # Buffer for the loss values
    loss_critic = torch.zeros(size=(args.num_optimizer_step,args.num_envs))
    loss_actor = torch.zeros(size=(args.num_optimizer_step,args.num_envs))
    for i in range(args.num_optimizer_step):

        # The actor
        ratio_pol = torch.exp(new_log_pol-log_pol)
        L = torch.min(advantage_GAE * ratio_pol , advantage_GAE*torch.clip(input=ratio_pol, min=1-args.clip_epsilon,max=1+args.clip_epsilon))
        expected_minus_L = -L.mean(dim=0).mean() # -L because AdamW minimize by default
        expected_minus_L.backward()

        # The critic
        value_loss = ((values-reward_to_go)**2).mean(dim=0).mean()
        value_loss.backward()

        # Running one optimizer step
        optimizer.step()
        optimizer.zero_grad()

        _,log_policy,entropy,value = agent.get_action_and_value(observations,actions,masks)
        values = agent.get_value(observations)
        new_log_pol = log_policy

        loss_critic[i] = value_loss.item()
        loss_actor[i] = expected_minus_L.item()

    heavy_args.logger.add_scalar("Loss/actor",loss_critic,heavy_args.logger_step)
    heavy_args.logger.add_scalar("Loss/critic",loss_actor,heavy_args.logger_step)
    heavy_args.logger.add_scalar("Loss/entropy",entropys.mean().mean().item(),heavy_args.logger_step)

    heavy_args.logger.add_scalar("Actor/total",scores[done].mean(dtype=torch.float).item(),heavy_args.logger_step)

    heavy_args.logger.add_scalar("Critic/value",value.mean().mean().item(),heavy_args.logger_step)
    heavy_args.logger.add_scalar("Critic/reward_to_go",reward_to_go.mean().mean().item(),heavy_args.logger_step)

    return observations, done, info
