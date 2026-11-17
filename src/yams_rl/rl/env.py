from yams_rl.rl.agent_player import AgentPlayer
from yams_rl.rl.encoding import build_observation, build_action_mask
from typing import Literal

import gymnasium as gym
import numpy as np

from yams_rl.rl.constant import (
    MAX_OPPONENTS,
    SCALE_DICE_SCORE,
    SCALE_PLAYER_GRID,
    SCALE_REROLL,
)
from yams_rl.rl.utils import nn_output_to_action
from yams_rl.yams.categories import Category
from yams_rl.yams.constant import NB_DICES, NB_MAX_ROLLS
from yams_rl.yams.dice import Dices
from yams_rl.yams.game_state import GameState
from yams_rl.yams.players import Player


class WrapperCartPole(gym.Wrapper):
    def __init__(self,cartpole_env:gym.Env):
        super().__init__(cartpole_env)
        self.cartpole = cartpole_env
        self.reward = 0

    def reset(self,seed:int|None=None,options:dict|None=None):
        self.reward = 0
        return self.cartpole.reset(seed=seed,options=options)

    def step(self,action):
        observation, reward, terminated, truncated, info = self.cartpole.step(action)
        self.reward += float(reward)
        info["agent_score"] = self.reward
        return observation.astype(np.float32), reward, terminated, truncated, info


class YamsEnv(gym.Env):
    def __init__(self,nb_players:int=1,reward_mode:Literal["delta","victory"]="delta",reward_norm_const:int=25,opponents_list:list|None=None):
        super().__init__()

        self.MAX_OPPONENTS = MAX_OPPONENTS
        assert nb_players<=self.MAX_OPPONENTS+1, f"The number of players is higher than the maximum : {self.MAX_OPPONENTS+1}"
        self.nb_players:int = nb_players
        self.dice_number:int = NB_DICES
        self.nb_max_rolls:int = NB_MAX_ROLLS
        self.reward_mode = reward_mode
        self.reward_norm_const = reward_norm_const

        # The agent can see :
            # the number of roll left
            # the dices
            # The scores available for each case for the current dices
            # his grid with a same size array indicating if the case is not filled yet (with 1 or 0)
            # how many players play
            # the grid of all the players with same size arrays for each players indicating if the case is not filled yet (1 or 0)
        self.size_input = (
            1 +
            self.dice_number +
            Category.size_category() +
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
        self.opponents_list = []
        self.opponents:dict[int,Player]

    def _set_state(self,state:GameState):
        self.state = state

    def action_mask(self):
        """Wrapper to builds the action mask for the nn output; True is a valid action"""
        return build_action_mask(state=self.state,dice_number=self.dice_number)

    def agent_score(self):
        """Return the agent current score, with potential Nan inside so no fake values can go through training unnoticed"""
        return self.state.total_player(player_id=self.agent_id)

    def _get_obs(self):
        """Wrapper to convert internal state to observation format.

        Returns:
            observation: np.ndarray
        """
        return build_observation(state=self.state,nb_players=self.nb_players,agent_id=self.agent_id)


    def reset(self,seed:int|None=None,options:dict|None=None):
        super().reset(seed=seed)
        seed = seed if seed else self.np_random_seed
        rng = np.random.default_rng(seed=seed)
        self.agent_id = options["agent_id"] if options and "agent_id" in options else int(rng.integers(low=0,high=self.nb_players))

        if self.opponents_list is not None:
            assert len(self.opponents_list)==self.nb_players-1, f"Number of opponents not matching the number of players: {len(self.opponents_list)} != {self.nb_players}-1"
            iter_opponents = iter(self.opponents_list)
            self.opponents = {op_id:next(iter_opponents) for op_id in range(self.nb_players) if op_id!=self.agent_id}


        for key in self.opponents:
            if isinstance(self.opponents[key],AgentPlayer):
                self.opponents[key].agent_id = key  # ty: ignore[unresolved-attribute]

        dices = Dices(dice_number=self.dice_number,seed=seed)
        state = GameState(players=self.nb_players,dices=dices,nb_max_roll=self.nb_max_rolls)

        self._set_state(state=state)

        # Playing until it's the agent turn
        while self.state.current_player_id != self.agent_id and not self.state.is_over():
            opponent_action = self.opponents[self.state.current_player_id].act(self.state)
            self.state.step(opponent_action)

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
        score_before = self.agent_score()
        self.state.step(actual_action)
        score_after = self.agent_score()

        while self.state.current_player_id != self.agent_id and not self.state.is_over():
            opponent_action = self.opponents[self.state.current_player_id].act(self.state)
            self.state.step(opponent_action)

        if self.reward_mode == "delta":
            reward = (score_after - score_before) / self.reward_norm_const # normalization
        else:
            reward = self.state.results(tie_value=0.5)[self.agent_id]

        observation = self._get_obs()

        terminated = self.state.is_over() # type is np.bool_
        truncated = False

        return observation, reward, bool(terminated), truncated, {"agent_score":self.agent_score()} #info
