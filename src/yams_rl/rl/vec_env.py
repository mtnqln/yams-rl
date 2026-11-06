import numpy as np
import torch
from gymnasium.spaces import Discrete

from yams_rl.rl.config import Args
from yams_rl.rl.env import YamsEnv


class SlowVecEnv:

    def __init__(self,envs:list[YamsEnv],generator:np.random.Generator,args:Args):
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

    def reset(self)->tuple[torch.Tensor,dict]:
        for idx,env in enumerate(self.envs):
            obs,_ = env.reset()
            self.observations[idx] = torch.from_numpy(obs)
        return (self.observations,{})

    def action_masks(self):
        for idx,env in enumerate(self.envs):
            self.masks[idx] = env.action_mask()

    def step(self,action:torch.Tensor|np.ndarray)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,dict]:
        """
        Run one step for each env in a python loop.
        Automatically reset the env where the game is over and send the first observation
        It returns observation, reward, terminated, truncated, info
        """
        for idx,env in enumerate(self.envs):
            observation, reward, terminated, truncated, info = env.step(int(action[idx].item()))
            self.observations[idx] = torch.from_numpy(observation)
            self.rewards[idx] = reward
            self.terminated[idx] = terminated
            self.truncated[idx] = truncated
            self.info["agents_scores"][idx] = info["agent_score"]

            if terminated or truncated:
                # in case of an end for an env we overwrite the last observation with the first one of the next game
                # The rewards etc will be taken care of the (1-done[t]) mask
                obs,_ = env.reset()
                self.observations[idx] = torch.from_numpy(obs)

        return self.observations,self.rewards,self.terminated,self.truncated,self.info
