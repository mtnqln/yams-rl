from __future__ import annotations

import numpy as np

from yams_rl.yams.categories import Action, Category
from yams_rl.yams.dice import Dices
from yams_rl.yams.scoring import score


class GameState:
    def __init__(self, players: int, nb_max_roll:int, dices:Dices):
        self.players_and_scores = np.full(fill_value=np.nan,shape=(players,len(Category)))
        self.players_number = players
        self.current_player_id = 0
        self.current_player_roll_number = 0
        self.nb_max_roll = nb_max_roll
        self.dices = dices
        self._start_turn()

    def _start_turn(self):
        self.dices.roll(reroll_mask=np.ones(self.dices.dice_number, dtype=bool))
        self.current_player_roll_number = 1

    def is_over(self):
        return ~np.isnan(self.players_and_scores).any()

    def scores(self,player_id:int,category:Category,dices:Dices):
        score_value = score(category=category, dices=dices)
        self.players_and_scores[player_id][category.idx] = score_value

    def legal_score(self)->list[Category]:
        grid = self.players_and_scores[self.current_player_id]
        legal_actions = Category.get_list()[np.isnan(grid)].tolist()
        return legal_actions

    def legal_score_mask(self)->np.ndarray:
        return np.isnan(self.players_and_scores[self.current_player_id])

    def reroll_possible(self)->bool:
        return self.current_player_roll_number < self.nb_max_roll

    def step(self,action:Action):

        if isinstance(action,np.ndarray):
            if self.current_player_roll_number < self.nb_max_roll:
                self.dices.roll(reroll_mask=action)
                self.current_player_roll_number+=1
            else:
                raise ValueError("No more dice roll possible")
        else:
            action = Category(action)
            if action not in self.legal_score():
                print(f"[ERROR] Action = {action} / legal move = {self.legal_score()}")
                raise ValueError("Illegal move")

            self.scores(player_id=self.current_player_id,category=action,dices=self.dices)
            self.current_player_id = (self.current_player_id+1)%self.players_number # modulus the number of player
            if not self.is_over():
                self._start_turn()

    def get_winner(self)->tuple[int,int]:
        highest_scores = self.players_and_scores.sum(axis=1)
        return highest_scores.max().astype(int),highest_scores.argmax().astype(int)

    def total_current_player(self):
        return np.nansum(self.players_and_scores[self.current_player_id])

    def total_player(self,player_id:int)->int:
        return int(np.nansum(self.players_and_scores[player_id]))

    def totals(self)->np.ndarray:
        """
        Returns the total for all players
        """
        return self.players_and_scores.sum(axis=1)
