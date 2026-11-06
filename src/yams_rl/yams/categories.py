from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

import numpy as np


class Category(StrEnum):
    UN = "1"
    DEUX = "2"
    TROIS = "3"
    QUATRE = "4"
    CINQ = "5"
    SIX = "6"
    BRELAN = "Brelan"
    CARRE = "Carre"
    FULL = "Full"
    PS = "PS"
    GS = "GS"
    YAMS = "Yams"
    PLUS = "+"
    MOINS = "-"

    @property
    @lru_cache
    def idx(self) -> int:
        return list(Category).index(self)

    @staticmethod
    @lru_cache
    def idx_from(action:int)->Category:
        return list(Category)[action]

    @staticmethod
    def get_list() -> np.ndarray:
        return np.array(Category,dtype=object)

    @staticmethod
    def size_category() -> int:
        return len(list(Category))

type Action = np.ndarray | Category
