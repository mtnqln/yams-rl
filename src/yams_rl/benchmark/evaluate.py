from collections import Counter

import numpy as np
import torch
import tyro
from joblib import Parallel, delayed
from numpy.random import SeedSequence

from yams_rl.rl.config import EvalArgs
from yams_rl.rl.model import Agent, load_model
from yams_rl.yams.constant import NB_DICES
from yams_rl.yams.dice import Dices
from yams_rl.yams.engine import play_one_game
from yams_rl.yams.player_factory import build_one_player
from yams_rl.yams.players import Player

"""
Baseline :
    RandomPlayer :
        Mean value from RandomPlayer = 26.68025
        Standard deviation from RandomPlayer = 17.411476385921443
        sigma/sqrt(N) = 0.12311773022954492
    Current agent trained on 368M steps (2026-11-16) :
        Stochastic policy:
            Mean value from Agent = 183.6876
            Standard deviation from Agent = 31.309334171138165
            sigma/sqrt(N) = 0.31309334171138165
        Deterministic policy:
            Mean value from Agent = 183.6361
            Standard deviation from Agent = 31.337793106567027
            sigma/sqrt(N) = 0.31337793106567025
    GreedyPlayer (size=5000):
        Mean value from GreedyPlayer = 210.3314
        Standard deviation from GreedyPlayer = 32.68697560252401
        sigma/sqrt(N) = 0.4622636421004792
"""


def evaluate(args:EvalArgs) -> tuple[float,float,float]|tuple[np.ndarray,np.ndarray,np.ndarray]:

    torch.set_num_threads(1)
    size = args.size_evaluation
    number_of_players = len(args.players)

    assert len(args.model_path)>= Counter(args.players)["Agent"], "You must provide a path for the weights of each agents"

    models = []
    idx=0
    for p in args.players:
        if p == "Agent":
            model = load_model(args.model_path[idx])
            assert isinstance(model, Agent)
            models.append(model)
            idx += 1
        else:
            models.append(None)

    def worker(i:int):
        sq = SeedSequence(entropy=i,spawn_key=[i+args.seed_offset])
        seeds = sq.spawn(number_of_players+3) # Number of players + pytorch + dices + permutations
        torch.manual_seed(int(seeds[1].generate_state(1).item()))
        if args.shuffle_seat:
            rng = np.random.default_rng(seed=seeds[-1].generate_state(1).item())
            permutations = rng.permutation(number_of_players).tolist()
        else:
            permutations = np.arange(number_of_players).tolist()
        players:list[Player] = build_players(args=args,seeds=seeds[2:],model=models,permutations=permutations)
        dices:Dices = Dices(dice_number=NB_DICES, seed=seeds[0])
        gamestate =  play_one_game(players=players, dices=dices)
        if number_of_players>1:
            results = np.empty(number_of_players)
            results[permutations] = gamestate.results(tie_value=0.5)
            return results
        else:
            return gamestate.totals()

    totals_counter = Parallel(n_jobs=-1)(
        delayed(worker)(i) for i in range(size)
    )

    array_totals_counter = np.array(totals_counter)
    if number_of_players>1:
        return array_totals_counter.mean(axis=0),array_totals_counter.std(axis=0),array_totals_counter.std(axis=0)/ np.sqrt(len(totals_counter))
    else:
        return array_totals_counter.mean(),array_totals_counter.std(),array_totals_counter.std()/ np.sqrt(len(totals_counter))


def build_players(args:EvalArgs,seeds:list[int]|list[SeedSequence],model:list[Agent|None],permutations:list[int])->list[Player]:
    players = []

    for idx in range(len(args.players)):
        players.append(build_one_player(
            player_type=args.players[permutations[idx]],
            seed=seeds[permutations[idx]],
            model=model[permutations[idx]],
            player_id=idx,
            stochastic=args.stochastic_policy))

    return players


if __name__=="__main__":
    eval_args = tyro.cli(EvalArgs)
    mean,std,std_err = evaluate(eval_args)

    if len(eval_args.players)==1:
        print(f"Mean value from {eval_args.players} = {mean}")
        print(f"Standard deviation from {eval_args.players} = {std}")
        print(f"sigma/sqrt(N) = {std_err}")
    else:
        assert isinstance(mean,np.ndarray)
        assert isinstance(std,np.ndarray)
        assert isinstance(std_err,np.ndarray)
        for idx,p in enumerate(eval_args.players):
            print(f"Winning rate from {p} = {mean[idx]}")
            print(f"sigma/sqrt(N) from {p} = {std_err[idx]}")
