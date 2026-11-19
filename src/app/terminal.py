from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
from numpy.random import SeedSequence

from yams_rl.rl.constant import MAX_OPPONENTS
from yams_rl.rl.model import Agent, load_model
from yams_rl.yams.categories import Action
from yams_rl.yams.constant import NB_DICES
from yams_rl.yams.dice import Dices, ManualDices
from yams_rl.yams.engine import play_one_game
from yams_rl.yams.game_state import GameState
from yams_rl.yams.player_factory import build_one_player
from yams_rl.yams.players import HumanPlayer, Player


@dataclass
class PlayArgs:
    players: tuple[str,...]
    """List of the players"""
    player_type: Literal["GreedyPlayer","Agent","RandomPlayer"] = "Agent"
    """Type of agent to play with"""
    stochastic_policy: bool = False
    """Whether to use or not stochastic policy for the agent"""
    model_path: str = ""
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
    players:list[Player] = [HumanPlayer(name=players_name[i],player_id=i) for i in range(number_players-1)]

    sq = SeedSequence(entropy=args.seed,spawn_key=[args.seed])
    seeds = sq.spawn(number_players)

    model = None
    if args.player_type=="Agent":
        assert args.model_path, "You must provide a path for loading the model"
        model = load_model(args.model_path)
        assert isinstance(model,Agent),"Model must be an Agent"
    agent = build_players(args,seeds[1:],model)

    players.insert(args.agent_id,agent)
    dices = ManualDices(dice_number=NB_DICES,seed=seeds[0]) if args.dices=="ManualDices" else Dices(dice_number=NB_DICES,seed=seeds[0])
    gamestate = play_one_game(players=players,dices=dices,logger_after=logger_after,logger_before=logger_before)

    totals = gamestate.totals().tolist()
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
    return build_one_player(args.player_type,seeds[0],model,args.agent_id,args.stochastic_policy)


if __name__=="__main__":
    args = tyro.cli(PlayArgs)
    play(args)
