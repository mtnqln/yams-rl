from yams_rl.yams.constant import NB_DICES
from yams_rl.yams.dice import Dices
from yams_rl.yams.engine import play_one_game
from yams_rl.yams.players import HumanPlayer, Player


def main() -> None:
    number_of_players=2
    players:list[Player] = [HumanPlayer(f"Player_{i}") for i in range(number_of_players)]
    dices = Dices(dice_number=NB_DICES,seed=25)
    play_one_game(players=players, dices=dices)

if __name__ == "__main__":
    main()
