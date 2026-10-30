from __future__ import annotations

import numpy as np

from yams_rl.yams.categories import Action, Category
from yams_rl.yams.dice import Dices
from yams_rl.yams.players import Player
from yams_rl.yams.scoring import score


class GameState:
    def __init__(self, players: list[Player],nb_max_roll:int,dices:Dices):
        self.players_and_scores = {player.name: np.full(fill_value=None, shape=(len(Category),)) for player in players}
        self.players = players
        self.players_number = len(players)
        self.current_player_id = 0
        self.current_player = players[0]
        self.current_player_roll_number = 0
        self.nb_max_roll = nb_max_roll
        self.dices = dices
        self._start_turn()

    def _start_turn(self):
        self.dices.roll(reroll_mask=np.ones(self.dices.dice_number, dtype=bool))
        self.current_player_roll_number = 1

    def is_over(self):
        return all(None not in player for player in self.players_and_scores.values())

    def scores(self,player:Player,category:Category,dices:Dices):
        score_value = score(category=category, dices=dices)
        self.players_and_scores[player.name][category.idx] = score_value

    def legal_score(self)->list[Category]:
        grid = self.players_and_scores[self.current_player.name]
        legal_actions = Category.get_list()[grid==None].tolist()
        return legal_actions

    def step(self,action:Action):

        if isinstance(action,np.ndarray):
            if self.current_player_roll_number < self.nb_max_roll:
                self.dices.roll(reroll_mask=action)
                self.current_player_roll_number+=1
            else:
                raise ValueError("No more dice roll possible")
        else:
            if action not in self.legal_score():
                raise ValueError("Illegal move")

            self.scores(player=self.current_player,category=action,dices=self.dices)
            self.current_player_id = (self.current_player_id+1)%self.players_number # modulus the number of player
            self.current_player = self.players[self.current_player_id]
            if not self.is_over():
                self._start_turn()

    def get_winner(self):
        highest_score = max(sum(list(pscore)) for pscore in self.players_and_scores.values())
        winner = list(self.players_and_scores.keys())[[sum(list(pscore)) for pscore in self.players_and_scores.values()].index(highest_score)]
        return winner

    def get_winner_score(self):
        return max(sum(list(pscore)) for pscore in self.players_and_scores.values())

    def totals(self):
        return [sum(list(pscore)) for pscore in self.players_and_scores.values()]
