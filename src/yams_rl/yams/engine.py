from collections.abc import Callable

import numpy as np

from src.yams_rl.yams.players import Player
from yams_rl.yams.constant import NB_MAX_ROLLS
from yams_rl.yams.game_state import Dices, GameState


def play_one_game(players: list[Player],dices:Dices,logger_before:Callable|None=None,logger_after:Callable|None=None)->list[int]:
    state:GameState = GameState(players=len(players),dices=dices,nb_max_roll=NB_MAX_ROLLS,start_turn=False)

    previous_player:int
    new_turn = True
    while not state.is_over():

        if logger_before and new_turn:
            logger_before(players=players,state=state)

        if new_turn:
            state._start_turn()

        previous_player = state.current_player_id
        action = players[state.current_player_id].act(state=state)
        state.step(action)

        if not isinstance(action,np.ndarray): # Not a reroll
            if logger_after:
                logger_after(players=players,previous_player=previous_player,action=action,state=state)
            new_turn = True
        else:
            new_turn = False

    return state.totals().tolist()
