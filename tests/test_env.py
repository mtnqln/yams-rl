from torch.ao.quantization.qconfig import default_per_channel_symmetric_qnnpack_qat_qconfig
import numpy as np
from gymnasium.utils.env_checker import check_env

from yams_rl.rl.env import YamsEnv
from yams_rl.rl.utils import action_to_nn_output
from yams_rl.yams.categories import Action
from yams_rl.yams.players import RandomPlayer


def test_env():

    env = YamsEnv(nb_players=2,reward_mode="delta")
    options = {"opponents":[RandomPlayer(name="P2",seed=25)]}
    obs,_ = env.reset(seed=10,options=options)
    a1 = RandomPlayer(name="a1",seed=10)
    over = False
    while not over:
        action:Action = a1.act(env.state)
        nn_like_action = action_to_nn_output(action)
        observation, reward, terminated, truncated, info = env.step(action=nn_like_action)
        over = terminated or truncated

    # This will catch many common issues
    env = YamsEnv(nb_players=1)
    check_env(env)
    print("Environment passes all checks!")

    assert True

def test_random_player_env():

    generator = np.random.default_rng(seed=50)
    values = []

    for i in range(10_000):
        seed:int = int(generator.integers(low=1,high=1_000_000))
        env = YamsEnv(nb_players=5,reward_mode="delta")
        options = {"agent_id":0,"opponents":[RandomPlayer(name=f"P{i}",player_id=i,seed=seed+i) for i in range(4)]}
        env.reset(seed=seed,options=options)
        a1 = RandomPlayer(name="a1",player_id=0,seed=seed+8)
        over = False
        while not over:
            action:Action = a1.act(env.state)
            nn_like_action = action_to_nn_output(action)
            observation, reward, terminated, truncated, info = env.step(action=nn_like_action)
            over = terminated or truncated

        values.append(env.state.total_player(player_id=0))

    values = np.array(values)
    assert 26.5 <= values.mean() <= 26.8
