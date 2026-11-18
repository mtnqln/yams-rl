from typing import Literal

from numpy.random import SeedSequence

from yams_rl.rl.agent_player import AgentPlayer
from yams_rl.yams.players import GreedyPlayer, RandomPlayer


def build_one_player(
    player_type: Literal["RandomPlayer", "Agent", "GreedyPlayer"],
    seed: int | SeedSequence,
    model=None,
    player_id:int=0,
    stochastic:bool=False):

    if player_type=="RandomPlayer":
        return RandomPlayer(f"Player_{player_id}",player_id=player_id,seed=seed)
    elif player_type=="Agent":
        assert model is not None,"No model provided for agent evaluation"
        return AgentPlayer(model=model,player_id=player_id,stochastic=stochastic)
    else:
        assert player_type=="GreedyPlayer"
        return GreedyPlayer(name=f"greedy-player-{player_id}",player_id=player_id)
