from dataclasses import dataclass
from typing import Literal


@dataclass
class Args:

    seed: int = 10
    """Seed for random reproducibility"""
    env_id: Literal["CartPole-v1","YamsEnv"] = "YamsEnv"
    """The id of the environment"""

    num_ppo_iter: int = 30_000
    """The number of ppo iteration"""
    num_step: int = 512
    """The number of steps to run in each environment per policy rollout"""
    num_envs: int = 16
    """The number of parallel game environments"""

    reward_norm_const: int = 25
    """Normalization constant for the reward"""

    num_minibatches: int = 4
    """the number of mini-batches"""
    num_epochs_optim: int = 8
    """the K epochs to update the policy"""
    clip_epsilon: float = 0.2
    """ppo clip coefficient"""
    gae_gamma: float = 1
    """GAE gamma coefficient"""
    gae_lambda: float = 0.98
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
    lr_decay_iter_start: int|None = 10_000
    """Number of iteration after which the lr decay starts"""
    lr_decay_min: float = 0
    """Minimum value to be reach by the lr decay"""
    entropy_decay_iter_start: int|None = 15_000
    """Number of iteration after which the entropy coeff decay starts"""
    entropy_decay_min: float = 0.0005
    """Minimum value for the entropy coef decay"""

    checkpoint_path: str|None = None
    """Checkpoint path"""

    opponents: tuple[Literal["RandomPlayer","Agent","GreedyPlayer"],...] = ()
    """Opponents to train the agent with; the agent means a frozen copy of the given agent with the path"""
    opponents_path: tuple[str,...] = ("models/agent.pt",)
    """Paths to the models weights selected for to train the agent"""

@dataclass
class EvalArgs:
    model_path: tuple[str,...] = ("models/agent.pt",)
    """Paths to the models weights selected for the 'players' option"""
    players: tuple[Literal["RandomPlayer","Agent","GreedyPlayer"],...] = ("RandomPlayer",)
    """Player to evaluate; more than one player means multiplayer game"""
    size_evaluation: int = 10_000
    """Number of evaluations to run"""
    stochastic_policy: bool = True
    """Whether to use stochastic policy or deterministic policy (for the neural network)"""
    seed_offset: int = 0
    """Offset to add to the seed generator key in order to change randomness for the eval"""
    shuffle_seat: bool = False
    """Whether to shuffle the seat order for the players"""
