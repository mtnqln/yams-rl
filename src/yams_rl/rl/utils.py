
from itertools import product
from typing import overload

import numpy as np

from yams_rl.yams.categories import Action, Category


def nn_to_dices_mapping(idx:int):
    mapping = list(product([0,1],repeat=5))

    return np.array(mapping[idx-Category.size_category()])

@overload
def nn_output_to_action(nn_action:int)->Action: ...

@overload
def nn_output_to_action(nn_action:np.ndarray)->np.ndarray: ...

def nn_output_to_action(nn_action:int|np.ndarray)->Action|np.ndarray:
    if isinstance(nn_action,np.ndarray):
        return Category.get_list()[nn_action]
    else:
        if Category.size_category() - 1 < nn_action:
            return nn_to_dices_mapping(nn_action)
        else:
            return Category.idx_from(nn_action)

# For testing purposes, useless in practice
def action_to_nn_output(action:Action)->int:
    size_grid = Category.size_category()

    conversion = [i for i in range(46)]

    if isinstance(action,Category):
        return conversion[action.idx]
    else:
        dices_comb = list(product([0,1],repeat=5))
        for idx,dice_comb in enumerate(dices_comb):
            if (action == np.array(dice_comb)).all():
                return idx+size_grid

    raise ValueError(f"action={action} is not of type Category or np.ndarray")
