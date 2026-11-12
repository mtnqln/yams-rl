import dataclasses
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from yams_rl.rl.config import Args
from yams_rl.rl.env import WrapperCartPole, YamsEnv
from yams_rl.rl.model import Agent
from yams_rl.rl.ppo import HeavyArgs, run_one_ppo_iter
from yams_rl.rl.vec_env import SlowVecEnv


def train_agent(args:Args):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/{args.env_id}_{timestamp}")
    generator = np.random.default_rng(seed=args.seed)
    seed = generator.integers(low=1,high=2**31)
    torch.manual_seed(seed)

    if args.env_id=="CartPole-v1":
        list_env = [WrapperCartPole(gym.make(args.env_id)) for _ in range(args.num_envs)]
    else:
        list_env = [YamsEnv(nb_players=args.num_player,reward_mode=args.reward_mode,reward_norm_const=args.reward_norm_const) for _ in range(args.num_envs)]

    envs = SlowVecEnv(envs=list_env,generator=generator,args=args)

    size_input = envs.single_observation_space.shape
    size_output = envs.single_action_space.n
    assert size_input is not None

    agent = Agent(size_input[0],size_output)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.optim_lr,
        eps=args.optim_eps,
        weight_decay=0 # using Adam
    )

    next_obs,_ = envs.reset()
    next_mask = envs.action_masks()
    next_done = torch.zeros(size=(args.num_envs,),dtype=torch.int)

    heavy_args = HeavyArgs(
            seed=generator,
            envs=envs,
            agent=agent,
            optimizer=optimizer,
            next_mask=next_mask,
            next_obs=next_obs,
            next_done=next_done,
            logger=writer,
            logger_step=0
        )

    if args.entropy_coef_decay:
        assert args.entropy_decay_iter_start<=args.num_ppo_iter
    if args.lr_decay:
        assert args.lr_decay_iter_start<=args.num_ppo_iter

    score = torch.zeros(size=(args.num_ppo_iter,))
    for i in range(args.num_ppo_iter):
        print(f"Run PPO {i+1}/{args.num_ppo_iter}")

        if args.lr_decay and i>=args.lr_decay_iter_start:
            lr_step = (args.optim_lr - args.lr_decay_min)/(args.num_ppo_iter-args.lr_decay_iter_start)
            heavy_args.optimizer.param_groups[0]["lr"] -= lr_step

        if args.entropy_coef_decay and i>=args.entropy_decay_iter_start:
            entropy_coef_step = (args.entropy_coef - args.entropy_decay_min)/(args.num_ppo_iter-args.lr_decay_iter_start)
            args.entropy_coef -= entropy_coef_step

        observations, done, mask, mean_score = run_one_ppo_iter(args=args,heavy_args=heavy_args)
        heavy_args.next_done = done[-1]
        heavy_args.next_obs = observations
        heavy_args.next_mask = mask
        score[i] = mean_score
        heavy_args.logger_step += args.num_step*args.num_envs # This counter allow to compare different runs with different num_step or num_envs

    save_dict = {
        "model_state_dict":agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "size_input":int(size_input[0]),
        "size_output":int(size_output),
        "training_config":dataclasses.asdict(args),
        "step_done":args.num_ppo_iter*(args.num_step*args.num_envs)

    }

    os.makedirs("models/",exist_ok=True)
    torch.save(save_dict,f"models/agent_{args.env_id}_{timestamp}.pt")

    # TODO: replace the hparam/score with a real evaluation metric
    writer.add_hparams(hparam_dict=dataclasses.asdict(args),metric_dict={"hparam/score":score.nanmean()})


if __name__== "__main__":
    args = tyro.cli(Args)
    train_agent(args)
