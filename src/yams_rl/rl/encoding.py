import numpy as np

from yams_rl.rl.constant import (
    MAX_OPPONENTS,
    SCALE_DICE_SCORE,
    SCALE_PLAYER_GRID,
    SCALE_REROLL,
)
from yams_rl.yams.categories import Category
from yams_rl.yams.constant import NB_MAX_ROLLS
from yams_rl.yams.game_state import GameState


def build_observation(state:GameState,nb_players:int,agent_id:int)->np.ndarray:
    """
    Convert the internal state to observation format.
    Returns:
        observation: np.ndarray
    """
    assert state.is_over() or state.current_player_id==agent_id

    rolls_left = NB_MAX_ROLLS - state.current_player_roll_number
    rolled_dices = state.dices.dices

    agent_grid = state.players_and_scores[agent_id].copy()
    agent_grid_mask = (~np.isnan(agent_grid)).astype(int)
    agent_grid[np.isnan(agent_grid)] = 0 # converting None to 0
    agent_grid = agent_grid.astype(int) # converting to int if it's not already the case

    opponents_playing = np.array([1 if i<nb_players-1 else 0 for i in range(MAX_OPPONENTS)])

    opponents_grids = np.delete(state.players_and_scores,agent_id,axis=0)
    # Filling up the non-playing player grid
    if len(opponents_grids) < MAX_OPPONENTS:
        diff = MAX_OPPONENTS - len(opponents_grids)
        filled_part_op_grid = np.full(fill_value=np.nan,shape=(diff,len(Category)))
        opponents_grids = np.concatenate((opponents_grids,filled_part_op_grid),axis=0)

    opponents_grids = np.array(opponents_grids)
    opponents_grids_masks = (~np.isnan(opponents_grids)).astype(int)
    opponents_grids[np.isnan(opponents_grids)] = 0 # converting None to 0
    opponents_grids = opponents_grids.astype(int) # from dtype=objet to dtype=int

    observation = np.concatenate((
        np.array([rolls_left])/SCALE_REROLL,
        rolled_dices/SCALE_DICE_SCORE,
        agent_grid/SCALE_PLAYER_GRID,
        agent_grid_mask,
        opponents_playing,
        opponents_grids.flatten()/SCALE_PLAYER_GRID,
        opponents_grids_masks.flatten()
    ))

    return observation.astype(np.float32)

def build_action_mask(state:GameState,dice_number:int):
    """
    Encode the legal actions for the agent to a boolean mask in order to switch its logits to -inf
    True is a valid action
    """
    legal_score = state.legal_score_mask()
    reroll = state.reroll_possible()
    dice_part = np.full(fill_value=reroll,shape=(2**dice_number,))
    scoring_part = legal_score

    return np.concat((scoring_part,dice_part)).astype(np.bool)
