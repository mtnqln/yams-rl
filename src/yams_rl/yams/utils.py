from yams_rl.yams.players import Player


def print_players_and_scores(players: list[Player], totals: list[int]) -> None:

    print("Player | Score")
    for player, total in zip(players, totals):
        print(player.name, total)
