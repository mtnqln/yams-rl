import numpy as np

from yams_rl.yams.engine import play_one_game
from yams_rl.yams.players import Player, RandomPlayer

"""
Baseline :
    RandomPlayer :
        Mean value from RandomPlayer = 26.68025
        Standard deviation from RandomPlayer = 17.411476385921443
        sigma/sqrt(N) = 0.12311773022954492
"""


def evaluate() -> None:

    size = 10_000

    number_of_players = 2
    dice_number = 5

    totals_counter = []
    counter = 0

    for i in range(size):
        sq = np.random.SeedSequence(entropy=i,spawn_key=[i])
        seeds = sq.spawn(number_of_players+1)
        players:list[Player] = [RandomPlayer(f"Player_{i}",seed=seeds[i]) for i in range(number_of_players)]
        totals =  play_one_game(players=players, dice_number=dice_number, seed=seeds[-1])
        totals_counter.extend(totals)
        counter += number_of_players

    print(f"Mean value from RandomPlayer = {np.array(totals_counter).mean()}")
    print(f"Standard deviation from RandomPlayer = {np.array(totals_counter).std()}")
    print(f"sigma/sqrt(N) = {np.array(totals_counter).std() / np.sqrt(counter)}")

if __name__=="__main__":
    evaluate()
