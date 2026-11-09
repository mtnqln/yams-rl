from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import SeedSequence

from yams_rl.yams.categories import Action, Category

if TYPE_CHECKING:
    from yams_rl.yams.game_state import GameState
from abc import ABC, abstractmethod
from itertools import product


class Player(ABC):
    name: str

    @abstractmethod
    def act(self, state: GameState) -> Action:
        pass

class HumanPlayer(Player):
    def __init__(self, name: str):
        self.name = name

    def act(self,state:GameState)->Action:
        if state.current_player_roll_number == state.nb_max_roll:
            action = self.read_from_player(state=state)
            while isinstance(action,np.ndarray):
                action = self.read_from_player(msg="no more dice roll possible",state=state)
            return action
        return self.read_from_player(state=state)

    def read_from_player(self,state:GameState,msg:str|None=None)->Action:
        if msg:
            print("\n",msg,"\n")
        print(f"\nCurrent dices = {state.dices.dices}")
        print("Roll dices again (d) or score (s)")
        choice = input()
        while choice != "d" and choice != "s":
            print("incorrect value")
            choice = input()

        if choice.strip() == "d":
            print("Choose which dice to roll again 1=roll 0=keep")
            dices = input()
            keep_dices = [dice=="1" for dice in dices]
            return np.array(keep_dices,dtype=bool)
        else:
            print(f"Choose a category to score from {[c.value for c in Category]}")
            while True:
                category=input()
                try:
                    return Category(category.strip())
                except ValueError:
                    print("incorrect value")


class RandomPlayer(Player):
    def __init__(self, name: str, seed:int|SeedSequence):
        self.name = name
        self.generator = np.random.default_rng(seed=seed)

    def act(self,state:GameState)->Action:
        return self.choose_random_action(state=state)

    def choose_random_action(self,state:GameState)->Action:
        legal_scores = state.legal_score()
        nb_roll = state.current_player_roll_number

        possible_actions = legal_scores
        if nb_roll < state.nb_max_roll:
            possible_actions = possible_actions + [np.array(reroll_mask,dtype=bool) for reroll_mask in product([True,False],repeat=state.dices.dice_number)]

        size_actions =  len(possible_actions)
        idx_action = self.generator.integers(low=0,high=size_actions)

        return possible_actions[idx_action]
