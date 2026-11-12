from collections import Counter
import numpy as np

from src.yams_rl.yams.categories import Category
from yams_rl.yams.dice import Dices


def check_legal_action(dices:Dices|np.ndarray,category:Category)->bool:
    dices = dices.get_dices().tolist() if isinstance(dices,Dices) else dices.tolist()
    count = Counter(dices)
    if category == "Brelan":
        return 3 in count.values() or 4 in count.values() or 5 in count.values()
    # "Full", "Carre", "Yams", "PS", "GS"]
    if category == "Full":
        return 3 in count.values() and 2 in count.values()
    if category == "Carre":
        return 4 in count.values() or 5 in count.values()
    if category == "Yams":
        return 5 in count.values()
    if category == "PS":
        return {1,2,3,4}.issubset(set(dices)) or {2,3,4,5}.issubset(set(dices)) or {3,4,5,6}.issubset(set(dices))
    if category == "GS":
        return {1,2,3,4,5}.issubset(set(dices)) or {2,3,4,5,6}.issubset(set(dices))
    return False


def score(category:Category, dices:Dices)-> int:
    if category == "1":
        return sum(d for d in dices.get_dices() if d == 1)
    elif category == "2":
        return sum(d for d in dices.get_dices() if d == 2)
    elif category == "3":
        return sum(d for d in dices.get_dices() if d == 3)
    elif category == "4":
        return sum(d for d in dices.get_dices() if d == 4)
    elif category == "5":
        return sum(d for d in dices.get_dices() if d == 5)
    elif category == "6":
        return sum(d for d in dices.get_dices() if d == 6)
    elif category == "+":
        return sum(dices.get_dices())
    elif category == "-":
        return -sum(dices.get_dices())
    elif category in ["Brelan", "Full", "Carre", "Yams", "PS", "GS"]:
        result = check_legal_action(dices, category)
        if result == False:
            return 0
        counts = Counter(dices.get_dices())
        if category == "Brelan":
            return counts.most_common(1)[0][0]*3
        elif category == "Full":
            return 30
        elif category == "Carre":
            return 40
        elif category == "Yams":
            return 50
        elif category == "PS":
            return 25
        elif category == "GS":
            return 35
    else:
        return 0

def score_arr(category:Category, dices:np.ndarray)-> int:
    if category == "1":
        return sum(d for d in dices if d == 1)
    elif category == "2":
        return sum(d for d in dices if d == 2)
    elif category == "3":
        return sum(d for d in dices if d == 3)
    elif category == "4":
        return sum(d for d in dices if d == 4)
    elif category == "5":
        return sum(d for d in dices if d == 5)
    elif category == "6":
        return sum(d for d in dices if d == 6)
    elif category == "+":
        return sum(dices)
    elif category == "-":
        return -sum(dices)
    elif category in ["Brelan", "Full", "Carre", "Yams", "PS", "GS"]:
        result = check_legal_action(dices, category)
        if result == False:
            return 0
        counts = Counter(dices)
        if category == "Brelan":
            return counts.most_common(1)[0][0]*3
        elif category == "Full":
            return 30
        elif category == "Carre":
            return 40
        elif category == "Yams":
            return 50
        elif category == "PS":
            return 25
        elif category == "GS":
            return 35
    else:
        return 0
