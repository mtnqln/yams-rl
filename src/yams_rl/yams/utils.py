from yams_rl.yams.players import Player


def print_winner(players: list[Player], totals: list[int]) -> None:

    print("Player | Score")
    for player, total in zip(players, totals):
        print(player.name, total)
