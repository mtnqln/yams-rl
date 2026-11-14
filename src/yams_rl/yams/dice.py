from typing import Literal
import numpy as np
from numpy.random import SeedSequence


class Dices:
    def __init__(self, dice_number:int, seed:int|SeedSequence):
        self.dice_number = dice_number
        self.dices = np.zeros(shape=(dice_number,),dtype=int)
        self.generator = np.random.default_rng(seed=seed)

    def get_dice(self, dice_number: int)->int:
        return self.dices[dice_number]
    def get_dices(self)->np.ndarray:
        return self.dices

    def roll(self,reroll_mask:np.ndarray):
        new_roll = self.generator.integers(low=1,high=7,size=self.dice_number)
        self.dices[reroll_mask] = new_roll[reroll_mask]
        self.dices.sort()

class ManualDices(Dices):
    def __init__(self,dice_number:int, seed:int|SeedSequence):
        super().__init__(dice_number, seed)

    def roll(self,reroll_mask:np.ndarray):
        if not reroll_mask.all():
            print(f"Please reroll the following dices: {self.dices[reroll_mask].tolist()}")
        else:
            print("Please roll all the dices")
        while True:
            print("Indicate the new scores: ")
            choices = input().strip()
            if len(choices)==int(reroll_mask.sum()) and all(c in "123456" for c in choices):
                break
            print(f"Enter exactly {int(reroll_mask.sum())} digits between 1 and 6 (for the dices that were rerolled")

        new_dices = [int(dice) for dice in choices]
        new_dices = np.array(new_dices)
        self.dices[reroll_mask] = new_dices
        self.dices.sort()
