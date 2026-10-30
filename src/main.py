from yams_rl.yams.engine import play_one_game
from yams_rl.yams.players import HumanPlayer, Player


def main() -> None:
    number_of_players=2
    players:list[Player] = [HumanPlayer(f"Player_{i}") for i in range(number_of_players)]
    play_one_game(players=players, dice_number=5,seed=10)

if __name__ == "__main__":
    main()
