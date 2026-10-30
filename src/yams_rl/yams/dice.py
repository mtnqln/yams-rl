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
