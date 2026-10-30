from enum import StrEnum

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
    def idx(self) -> int:
        return list(Category).index(self)

    @staticmethod
    def get_list() -> np.ndarray:
        return np.array(list(Category))

type Action = np.ndarray | Category
