from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:

    seed: int = 10
    """Seed for random reproducibility"""
    num_ppo_iter: int = 15
    """The number of ppo iteration"""
    num_step: int = 500
    """The number of steps to run in each environment per policy rollout"""
    num_envs: int = 1
    """The number of parallel game environments"""
    num_player: int = 1
    """Number of player per environment; currently only support one"""
    reward_mode: Literal["delta","victory"] = "delta"
    """Reward mode to train the agent, victory is 1=win 0=loose; delta is the difference of score between each round"""
    num_optimizer_step: int = 5
    """The number of optimizer steps to run per ppo update"""
    clip_epsilon: float = 0.2
    """ppo clip coefficient"""
    gae_gamma: float = 0.96
    """GAE gamma coefficient"""
    gae_lambda: float = 0.96
    """GAE lambda coefficient"""

    optim_lr: float = 2.5e-4
    """Optimizer learning rate"""
    optim_eps: float = 4.3
    """Optimizer (AdamW) epsilon"""
