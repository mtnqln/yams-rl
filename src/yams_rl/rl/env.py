from typing import Literal
from yams_rl.yams.categories import Action
from yams_rl.yams.dice import Dices
import numpy as np
import gymnasium as gym
from numpy.random import SeedSequence

from utils import get_config
from yams_rl.yams.game_state import GameState
from yams_rl.yams.players import Player, AgentPlayer

class YamsEnv(gym.Env):
    def __init__(self,nb_players:int=1,reward_mode:Literal["delta","victory"]="delta"):
        super().__init__()

        config = get_config()
        MAX_OPPONENTS = config["env"]["MAX_OPPONENTS"]
        SIZE_OUTPUT = config["nn"]["SIZE_OUTPUT"]
        NB_DICES = config["env"]["NB_DICES"]
        NB_MAX_ROLLS = config["env"]["NB_MAX_ROLLS"]

        self.nb_players:int = nb_players
        self.dice_number:int = NB_DICES
        self.nb_max_rolls:int = NB_MAX_ROLLS
        self.reward_mode = reward_mode

        # The agent can see :
            # the number of roll left
            # the dices
            # his grid
            # the grid of all the players
        self.observation_space = gym.spaces.MultiDiscrete(nvec=MAX_OPPONENTS)
        self.action_space = gym.spaces.Discrete(SIZE_OUTPUT)

        self.agent_name = "a1"

        self._state:GameState

    def _set_state(self,state:GameState):
        self._state = state

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation
        """

        rolled_dices = self._state.dices.dices
        possible_score = self._state.legal_score()
        agent_grid = self._state.players_and_scores[self._state.current_player.name]
        opponents_playing = [i for i in range(self.nb_players - 1)]
        opponents_grids = (
            [self._state.players_and_scores[opponent.name]
            for opponent in self._state.players
            if opponent.name != self._state.current_player.name]
        )

        observation = np.concatenate((
            rolled_dices,
            possible_score,
            agent_grid,
            opponents_playing,
            opponents_grids
        ))

        return observation

    def reset(self,seed:int|None=None,options:dict|None=None):
        super().reset(seed=seed)
        assert isinstance(seed,int)
        assert isinstance(options,dict)

        opponent_players:list[Player] = options["opponent_players"]
        agent_player = AgentPlayer(self.agent_name,seed=seed)
        players:list[Player] = opponent_players + [agent_player]
        dices = Dices(dice_number=self.dice_number,seed=seed)
        state = GameState(players=players,dices=dices,nb_max_roll=self.nb_max_rolls)

        self._set_state(state=state)
        observation = self._get_obs()

        return observation

    def step(self,action:Action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        score_before = self._state.total()
        self._state.step(action)
        score_after = self._state.total()

        if self.reward_mode == "delta":
            reward = (score_after - score_before) / score_after # normalization
        else:
            reward = 1 if self._state.is_over() and self._state.get_winner() == self.agent_name else 0

        observation = self._get_obs()

        terminated = self._state.is_over()
        truncated = False

        return observation, reward, terminated, truncated #, info
