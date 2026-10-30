from yams_rl.yams.engine import play_one_game


def main() -> None:
    play_one_game(number_of_players=2, dice_number=5,seed=10)

if __name__ == "__main__":
    main()
