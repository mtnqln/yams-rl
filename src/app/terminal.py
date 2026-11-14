from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
from numpy.random import SeedSequence

from yams_rl.rl.agent_player import AgentPlayer
from yams_rl.rl.constant import MAX_OPPONENTS
from yams_rl.rl.model import Agent, load_model
from yams_rl.yams.categories import Action
from yams_rl.yams.constant import NB_DICES
from yams_rl.yams.dice import Dices, ManualDices
from yams_rl.yams.engine import play_one_game
from yams_rl.yams.game_state import GameState
from yams_rl.yams.players import GreedyPlayer, HumanPlayer, Player, RandomPlayer
from yams_rl.yams.scoring import score


@dataclass
class PlayArgs:
    players: tuple[str,...]
    """List of the players"""
    player_type: Literal["GreedyPlayer","Agent","RandomPlayer"] = "Agent"
    """Type of agent to play with"""
    stochastic_policy: bool = False
    """Whether to use or not stochastic policy for the agent"""
    model_path: str = "models/agent.pt"
    """Path to the agent weights"""
    seed: int = 50
    """Seed for random reproducibility"""
    agent_id: int = 0
    """Agent position in the round"""
    dices: Literal["Dices","ManualDices"] = "ManualDices"

def play(args:PlayArgs):
    players_name = args.players
    number_players = len(players_name) + 1
    assert number_players-1<=MAX_OPPONENTS, "Too many players"
    assert args.agent_id < number_players, f"Agent position must be within [0,{number_players})"
    players:list[Player] = [HumanPlayer(name=players_name[i]) for i in range(number_players-1)]

    sq = SeedSequence(entropy=args.seed,spawn_key=[args.seed])
    seeds = sq.spawn(number_players)

    model = None
    if args.player_type=="Agent":
        model = load_model(args.model_path)
    agent = build_players(args,seeds[1:],model)

    players.insert(args.agent_id,agent)
    dices = ManualDices(dice_number=NB_DICES,seed=seeds[0]) if args.dices=="ManualDices" else Dices(dice_number=NB_DICES,seed=seeds[0])
    totals = play_one_game(players=players,dices=dices,logger_after=logger_after,logger_before=logger_before)

    for t,p in zip(totals,players):
        print(f"Player {p.name} has a total of {t}")

def logger_before(players:list[Player], state:GameState):
    print(f"\nPlayer {players[state.current_player_id].name} to play")
    print(f"Current player grid is : {state.players_and_scores[state.current_player_id]}")

def logger_after(players:list[Player],previous_player:int,state:GameState,action:Action):
    if isinstance(action,np.ndarray):
        print(f"Player {players[previous_player].name} relaunched his dices")
    else:
        print(f"\nPlayer {players[previous_player].name} scored {state.players_and_scores[previous_player][action.idx]} with {action}")
        print(f"Current player grid is : {state.players_and_scores[previous_player]}")

def build_players(args:PlayArgs,seeds:list[int]|list[SeedSequence],model:Agent|None=None)->Player:
    if args.player_type=="RandomPlayer":
        return RandomPlayer(f"Player_{args.agent_id}",seed=seeds[0])
    elif args.player_type=="Agent":
        assert model is not None,"No model provided for agent evaluation"
        return AgentPlayer(model=model,agent_id=args.agent_id,stochastic=args.stochastic_policy)
    else:
        assert args.player_type=="GreedyPlayer"
        return GreedyPlayer(name=f"greedy-player-{args.agent_id}")


if __name__=="__main__":
    args = tyro.cli(PlayArgs)
    play(args)
