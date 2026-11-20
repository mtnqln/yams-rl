from __future__ import annotations
from yams_rl.yams.scoring import score, score_arr
from collections import Counter
from yams_rl.yams.dice import Dices
from numpy.lib.stride_tricks import sliding_window_view
from math import factorial

from functools import cache, lru_cache
from typing import TYPE_CHECKING, Iterable

import numpy as np
from numpy.random import SeedSequence

from yams_rl.yams.constant import NB_DICES
from yams_rl.yams.categories import Action, Category

if TYPE_CHECKING:
    from yams_rl.yams.game_state import GameState
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement, product


class Player(ABC):
    name: str
    player_id: int

    @abstractmethod
    def act(self, state: GameState) -> Action:
        pass

class HumanPlayer(Player):
    def __init__(self, name: str, player_id: int):
        self.name = name
        self.player_id = player_id

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
            while True:
                print("Choose which dice to roll again 1=roll 0=keep")
                dices = input()
                if len(dices)==NB_DICES and all(dice=="1" or dice=="0" for dice in dices):
                    break
                else:
                    print(f"Incorrect input make sure all numbers are either 0 or 1 and the lenght is {NB_DICES}")
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
    def __init__(self, name: str, player_id: int, seed:int|SeedSequence):
        self.player_id = player_id
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


class GreedyPlayer(Player):
    def __init__(self, name: str, player_id:int):
        self.player_id = player_id
        self.name = name

    def act(self,state:GameState)->Action:
        grid = state.players_and_scores[state.current_player_id]
        valid_action:np.ndarray = np.isnan(grid)
        if state.current_player_roll_number >= state.nb_max_roll:
            valid_score = score_once(tuple(state.dices.dices.tolist()),tuple(valid_action.tolist()))
            return state.legal_score()[np.array(valid_score).argmax()]
        else:
            expectations = np.array(compute_expectations(dices=tuple(state.dices.dices.tolist()),dice_number=state.dices.dice_number,valid_action=tuple(valid_action.tolist())))
            if max(score_once(tuple(state.dices.dices.tolist()),tuple(valid_action.tolist())))>= expectations.max():
                valid_score = score_once(tuple(state.dices.dices.tolist()),tuple(valid_action.tolist()))
                return state.legal_score()[np.array(valid_score).argmax()]
            else:
                return build_masks(state.dices.dice_number)[expectations.argmax()]

def build_masks(dice_number:int)->np.ndarray:
    return np.array([np.array(reroll_mask,dtype=bool) for reroll_mask in product([True,False],repeat=dice_number)])

@cache
def score_once(d:tuple,vc:tuple)->tuple:
    dices = np.array(d)
    valid_category = np.array(vc)
    scores = []
    for c in np.array(Category.get_list())[valid_category]:
        scores.append(score_arr(c,dices))
    return tuple(scores)

@cache
def compute_expectations(dices:tuple,dice_number:int,valid_action:tuple)->list[float]:
    masks = build_masks(dice_number=dice_number)
    expected_val = []
    for m in masks:
        rerolls = m.sum()
        new_rolls = product([1,2,3,4,5,6],repeat=rerolls)
        exp = []
        dices_arr = np.array(dices)
        for p in new_rolls:
            new_dices = dices_arr.copy()
            new_dices[m] = p
            new_dices.sort()
            exp.append((1/6)**rerolls * max(score_once(tuple(new_dices.tolist()),valid_action)))
        expected_val.append(np.array(exp).sum())
    return expected_val
