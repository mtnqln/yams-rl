import numpy as np
import torch
import tyro
from joblib import Parallel, delayed
from numpy.random import SeedSequence

from yams_rl.rl.agent_player import AgentPlayer
from yams_rl.rl.config import EvalArgs
from yams_rl.rl.model import Agent, load_model
from yams_rl.yams.constant import NB_DICES
from yams_rl.yams.dice import Dices
from yams_rl.yams.engine import play_one_game
from yams_rl.yams.players import GreedyPlayer, Player, RandomPlayer

"""
Baseline :
    RandomPlayer :
        Mean value from RandomPlayer = 26.68025
        Standard deviation from RandomPlayer = 17.411476385921443
        sigma/sqrt(N) = 0.12311773022954492
    Current agent trained on 60_000 steps (2026-11-11) :
        Stochastic policy:
            Mean value from Agent = 169.3563
            Standard deviation from Agent = 28.831960569999396
            sigma/sqrt(N) = 0.28831960569999393
        Deterministic policy:
            Mean value from Agent = 169.4747
            Standard deviation from Agent = 29.07117403735185
            sigma/sqrt(N) = 0.2907117403735185
    GreedyPlayer (size=5000):
        Mean value from GreedyPlayer = 210.3314
        Standard deviation from GreedyPlayer = 32.68697560252401
        sigma/sqrt(N) = 0.4622636421004792
"""


def evaluate(args:EvalArgs) -> tuple[float,float,float]:

    torch.set_num_threads(1)
    size = args.size_evaluation
    number_of_players = args.number_players

    model = None
    if args.player_type=="Agent":
        model = load_model(args.model_path)

    def worker(i:int):
        sq = SeedSequence(entropy=i,spawn_key=[i+args.seed_offset])
        seeds = sq.spawn(number_of_players+2) # Number of players + pytorch + dices
        torch.manual_seed(int(seeds[1].generate_state(1).item()))
        players:list[Player] = build_players(args=args,seeds=seeds[2:],model=model)
        dices:Dices = Dices(dice_number=NB_DICES, seed=seeds[0])
        totals =  play_one_game(players=players, dices=dices)
        return totals

    totals_counter = Parallel(n_jobs=-1)(
        delayed(worker)(i) for i in range(size)
    )

    array_totals_counter = np.array(totals_counter)
    return array_totals_counter.mean(),array_totals_counter.std(),array_totals_counter.std()/ np.sqrt(len(totals_counter))


def build_players(args:EvalArgs,seeds:list[int]|list[SeedSequence],model:Agent|None=None)->list[Player]:

    if args.player_type=="RandomPlayer":
        assert args.number_players==len(seeds),"The number of RandomPlayer is not matching the number of seeds"
        return [RandomPlayer(f"Player_{i}",seed=seeds[i]) for i in range(args.number_players)]
    elif args.player_type=="Agent":
        assert model is not None,"No model provided for agent evaluation"
        assert args.number_players==1,"Agent evaluation for multiple agents is not implemented"
        return [AgentPlayer(model=model,agent_id=0,stochastic=args.stochastic_policy)]
    else:
        assert args.player_type=="GreedyPlayer"
        return [GreedyPlayer(name="greedy-player-0")]



if __name__=="__main__":
    eval_args = tyro.cli(EvalArgs)
    mean,std,std_err = evaluate(eval_args)

    print(f"Mean value from {eval_args.player_type} = {mean}")
    print(f"Standard deviation from {eval_args.player_type} = {std}")
    print(f"sigma/sqrt(N) = {std_err}")
