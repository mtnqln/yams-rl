from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:

    seed: int = 10
    """Seed for random reproducibility"""
    env_id: Literal["CartPole-v1","YamsEnv"] = "YamsEnv"
    """The id of the environment"""

    num_ppo_iter: int = 60_000
    """The number of ppo iteration"""
    num_step: int = 256
    """The number of steps to run in each environment per policy rollout"""
    num_envs: int = 4
    """The number of parallel game environments"""
    num_player: int = 1
    """Number of player per environment; currently only support one"""

    reward_mode: Literal["delta","victory"] = "delta"
    """Reward mode to train the agent for the Yams environment, victory is 1=win 0=loose; delta is the difference of score between each round"""
    reward_norm_const: int = 25
    """Normalization constant for the reward"""

    num_optimizer_step: int = 10
    """The number of optimizer steps to run per ppo update"""
    clip_epsilon: float = 0.2
    """ppo clip coefficient"""
    gae_gamma: float = 1
    """GAE gamma coefficient"""
    gae_lambda: float = 1
    """GAE lambda coefficient"""
    norm_adv: bool = True
    """Wether to use advantages normalization"""
    max_gradient_norm: float = 0.5
    """The maximum norm for the gradient clipping"""
    optim_lr: float = 2.5e-4
    """Optimizer learning rate"""
    optim_eps: float = 1e-5
    """Optimizer (AdamW) epsilon"""
    entropy_coef: float = 0.01
    """Coefficient for the entropy term in the loss"""
    lr_decay: bool = True
    """Toggle the decay of the learning rate for the optimizer"""
    lr_decay_iter_start: int = 30_000
    """Number of iteration after which the lr decay starts"""
    lr_decay_min: float = 1e-6
    """Minimum value to be reach by the lr decay"""
    entropy_coef_decay: bool = True
    """Toggle the decay of the entropy coeff"""
    entropy_decay_iter_start: int = 30_000
    """Number of iteration after which the entropy coeff decay starts"""
    entropy_decay_min: float = 0.0005
    """Minimum value for the entropy coef decay"""

@dataclass
class EvalArgs:
    model_path: str = "models/agent.pt"
    """Path to the model weights"""
    player_type: Literal["RandomPlayer","Agent","GreedyPlayer"] = "RandomPlayer"
    """Type of player to evaluate"""
    size_evaluation: int = 10_000
    """Number of evaluations to run"""
    number_players: int = 1
    """Number of player to put in the evaluation"""
    stochastic_policy: bool = True
    """Whether to use stochastic policy or deterministic policy (for the neural network)"""
    seed_offset: int = 0
    """Offset to add to the seed generator key in order to change randomness for the eval"""
