from numpy.random import SeedSequence

from src.yams_rl.yams.players import Player
from yams_rl.yams.game_state import Dices, GameState


def play_one_game(players: list[Player],dice_number:int,seed:int|SeedSequence)->list[int]:
    dices:Dices = Dices(dice_number=dice_number, seed=seed)
    state:GameState = GameState(players=len(players),dices=dices,nb_max_roll=3)

    while not state.is_over():
        action = players[state.current_player_id].act(state=state)
        state.step(action)

    return state.totals().tolist()
