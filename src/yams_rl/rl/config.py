from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:

    seed: int = 10
    """Seed for random reproducibility"""
    env_id: Literal["CartPole-v1","YamsEnv"] = "YamsEnv"
    """The id of the environment"""
    num_ppo_iter: int = 15_000
    """The number of ppo iteration"""
    num_step: int = 256
    """The number of steps to run in each environment per policy rollout"""
    num_envs: int = 4
    """The number of parallel game environments"""
    num_player: int = 1
    """Number of player per environment; currently only support one"""
    reward_mode: Literal["delta","victory"] = "delta"
    """Reward mode to train the agent for the Yams environment, victory is 1=win 0=loose; delta is the difference of score between each round"""
    num_optimizer_step: int = 10
    """The number of optimizer steps to run per ppo update"""
    clip_epsilon: float = 0.2
    """ppo clip coefficient"""
    gae_gamma: float = 1
    """GAE gamma coefficient"""
    gae_lambda: float = 0.95
    """GAE lambda coefficient"""
    norm_adv: bool = True
    """Wether to use advantages normalization"""
    max_gradient_norm: float = 0.5
    """The maximum norm for the gradient clipping"""
    optim_lr: float = 2.5e-4
    """Optimizer learning rate"""
    optim_eps: float = 1e-5
    """Optimizer (AdamW) epsilon"""
