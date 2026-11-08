from copy import deepcopy
import collections

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Discrete

from yams_rl.rl.config import Args


class SlowVecEnv:

    def __init__(self,envs:collections.abc.Sequence[gymnasium.Env],generator:np.random.Generator,args:Args):
        self.envs = envs
        self.single_observation_space = self.envs[0].observation_space
        assert isinstance(self.envs[0].action_space,Discrete)
        self.single_action_space = self.envs[0].action_space

        # Buffers
        self.masks = torch.ones(size=(args.num_envs,self.single_action_space.n) ,dtype=torch.bool)
        # (a,) + (b,) = (a,b)
        self.observations = torch.zeros(size=(args.num_envs,)+self.single_observation_space.shape)  # ty: ignore[unsupported-operator]
        self.rewards = torch.zeros(size = (args.num_envs,))
        self.terminated = torch.zeros_like(self.rewards,dtype=torch.bool)
        self.truncated = torch.zeros_like(self.rewards,dtype=torch.bool)
        self.info = {"agents_scores":torch.zeros_like(self.rewards)}

        self.generator = generator

    def reset(self)->tuple[torch.Tensor,dict]:
        for idx,env in enumerate(self.envs):
            seed = int(self.generator.integers(low=1,high=2**31))
            obs,_ = env.reset(seed=seed)
            self.observations[idx] = torch.from_numpy(obs)
        return (self.observations.clone(),{})

    def action_masks(self)->torch.Tensor:
        if hasattr(self.envs[0],"action_mask") and callable(self.envs[0].action_mask):
            for idx,env in enumerate(self.envs):
                self.masks[idx] = torch.from_numpy(env.action_mask())  # ty: ignore[unresolved-attribute]
            return self.masks.clone()
        else:
            return torch.full_like(self.masks,fill_value=True)


    def step(self,action:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,dict]:
        """
        Run one step for each env in a python loop.
        Automatically reset the env where the game is over and send the first observation
        It returns observation, reward, terminated, truncated, info
        """
        for idx,env in enumerate(self.envs):
            observation, reward, terminated, truncated, info = env.step(int(action[idx].item()))
            self.observations[idx] = torch.from_numpy(observation).to(torch.float)
            self.rewards[idx] = float(reward)
            self.terminated[idx] = terminated
            self.truncated[idx] = truncated
            self.info["agents_scores"][idx] = info["agent_score"]

            if terminated or truncated:
                # in case of an end for an env we overwrite the last observation with the first one of the next game
                # The rewards etc will be taken care of the (1-done[t]) mask
                seed = int(self.generator.integers(low=1,high=2**31))
                obs,_ = env.reset(seed=seed)
                self.observations[idx] = torch.from_numpy(obs)

        return self.observations.clone(),self.rewards.clone(),self.terminated.clone(),self.truncated.clone(),deepcopy(self.info)
