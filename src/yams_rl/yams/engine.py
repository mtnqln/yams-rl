from src.yams_rl.yams.players import Player
from yams_rl.yams.game_state import Dices, GameState
from yams_rl.yams.players import HumanPlayer


def play_one_game(number_of_players: int,dice_number:int,seed:int)->list[int]:
    players:list[Player] = [HumanPlayer(f"Player_{i}") for i in range(number_of_players)]
    dices:Dices = Dices(dice_number=dice_number, seed=seed)
    state:GameState = GameState(players=players,dices=dices,nb_max_roll=3)

    while not state.is_over():
        action = state.current_player.act(state=state)
        state.step(action)

    return state.totals()
