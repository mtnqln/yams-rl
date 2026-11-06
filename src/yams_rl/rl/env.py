from typing import Literal

import gymnasium as gym
import numpy as np
import torch

from yams_rl.rl.constant import MAX_OPPONENTS
from yams_rl.rl.utils import nn_output_to_action
from yams_rl.yams.categories import Category
from yams_rl.yams.constant import NB_DICES, NB_MAX_ROLLS
from yams_rl.yams.dice import Dices
from yams_rl.yams.game_state import GameState
from yams_rl.yams.players import Player


class YamsEnv(gym.Env):
    def __init__(self,nb_players:int=1,reward_mode:Literal["delta","victory"]="delta"):
        super().__init__()

        self.MAX_OPPONENTS = MAX_OPPONENTS
        assert nb_players<self.MAX_OPPONENTS, f"The number of players is higher than the maximum : {self.MAX_OPPONENTS}"
        self.nb_players:int = nb_players
        self.dice_number:int = NB_DICES
        self.nb_max_rolls:int = NB_MAX_ROLLS
        self.reward_mode = reward_mode
        self.normalization_const = 50

        # The agent can see :
            # the number of roll left
            # the dices
            # his grid with a same size array indicating if the case is not filled yet (with 1 or 0)
            # how many players play
            # the grid of all the players with same size arrays for each players indicating if the case is not filled yet (1 or 0)
        self.size_input = (
            1 +
            self.dice_number +
            2*Category.size_category() +
            MAX_OPPONENTS +
            2*MAX_OPPONENTS * Category.size_category()

        )

        self.size_output = (
            Category.size_category() +
            2**self.dice_number
        )

        self.observation_space = gym.spaces.Box(low=-500,high=500,shape=(self.size_input,))
        self.action_space = gym.spaces.Discrete(self.size_output)

        # Will be initialized with .reset()
        self.state:GameState
        self.agent_id:int
        self.opponents:dict[int,Player]

    def _set_state(self,state:GameState):
        self.state = state

    def action_mask(self):
        """Builds the action mask for the nn output; True is a valid action"""
        legal_score = self.state.legal_score_mask()
        reroll = self.state.reroll_possible()

        dice_part = torch.full(fill_value=reroll,size=(2**self.dice_number,))
        scoring_part = torch.from_numpy(legal_score)

        return torch.concat((scoring_part,dice_part)).to(torch.bool)

    def agent_score(self):
        """Return the agent current score, with potential Nan inside so no fake values can go through training unnoticed"""
        return self.state.total_player(player_id=self.agent_id)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            observation: np.ndarray
        """

        assert self.state.is_over() or self.state.current_player_id==self.agent_id

        rolls_left = self.nb_max_rolls - self.state.current_player_roll_number
        rolled_dices = self.state.dices.dices

        agent_grid = self.state.players_and_scores[self.agent_id].copy()
        agent_grid_mask = (~np.isnan(agent_grid)).astype(int)
        agent_grid[np.isnan(agent_grid)] = 0 # converting None to 0
        agent_grid = agent_grid.astype(int) # converting to int if it's not already the case

        opponents_playing = np.array([1 if i<self.nb_players-1 else 0 for i in range(self.MAX_OPPONENTS)])

        opponents_grids = np.delete(self.state.players_and_scores,self.state.current_player_id,axis=0)
        # Filling up the non-playing player grid
        if len(opponents_grids) < self.MAX_OPPONENTS:
            diff = self.MAX_OPPONENTS - len(opponents_grids)
            filled_part_op_grid = np.full(fill_value=np.nan,shape=(diff,len(Category)))
            opponents_grids = np.concatenate((opponents_grids,filled_part_op_grid),axis=0)

        opponents_grids = np.array(opponents_grids)
        opponents_grids_masks = (~np.isnan(opponents_grids)).astype(int)
        opponents_grids[np.isnan(opponents_grids)] = 0 # converting None to 0
        opponents_grids = opponents_grids.astype(int) # from dtype=objet to dtype=int

        observation = np.concatenate((
            np.array([rolls_left]),
            rolled_dices,
            agent_grid,
            agent_grid_mask,
            opponents_playing,
            opponents_grids.flatten(),
            opponents_grids_masks.flatten()
        ))

        return observation.astype(np.float32)

    def reset(self,seed:int|None=None,options:dict|None=None):
        super().reset(seed=seed)
        seed = seed if seed else self.np_random_seed
        self.agent_id = options["agent_id"] if options and "agent_id" in options else 0

        if options and "opponents" in options:
            assert len(options["opponents"])==self.nb_players-1, f"Number of opponents not matching the number of players: {len(options["opponents"])} != {self.nb_players}-1"
            iter_opponents = iter(options["opponents"])
            self.opponents = {op_id:next(iter_opponents) for op_id in range(self.nb_players) if op_id!=self.agent_id}
        else:
            self.opponents = {}

        dices = Dices(dice_number=self.dice_number,seed=seed)
        state = GameState(players=self.nb_players,dices=dices,nb_max_roll=self.nb_max_rolls)

        self._set_state(state=state)
        observation = self._get_obs()

        return observation,{} #info


    def step(self,action:int):
        """Execute one timestep within the environment.

        Args:
            nn_action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        actual_action = nn_output_to_action(action)
        score_before = self.state.total_current_player()
        self.state.step(actual_action)
        score_after = self.state.total_current_player()

        if self.reward_mode == "delta":
            reward = (score_after - score_before) / self.normalization_const # normalization
        else:
            reward = 1 if self.state.is_over() and self.state.get_winner()[1] == self.agent_id else 0 # case of equality to be hanble yet

        while self.state.current_player_id != self.agent_id and not self.state.is_over():
            opponent_action = self.opponents[self.state.current_player_id].act(self.state)
            self.state.step(opponent_action)

        observation = self._get_obs()

        terminated = self.state.is_over()
        truncated = False

        return observation, reward, terminated, truncated, {"agent_score":self.agent_score()} #info
