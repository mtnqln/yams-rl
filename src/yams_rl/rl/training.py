import dataclasses

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from yams_rl.rl.config import Args
from yams_rl.rl.env import YamsEnv
from yams_rl.rl.model import Agent
from yams_rl.rl.ppo import HeavyArgs, run_one_ppo_iter
from yams_rl.rl.vec_env import SlowVecEnv


def train_agent(args:Args):

    writer = SummaryWriter("runs/yams_rl1")
    generator = np.random.default_rng(seed=args.seed)


    list_yams_env = [YamsEnv(nb_players=args.num_player,reward_mode=args.reward_mode) for i in range(args.num_envs)]
    envs = SlowVecEnv(envs=list_yams_env,generator=generator,args=args)

    size_input = list_yams_env[0].size_input
    size_output = list_yams_env[0].size_output

    agent = Agent(size_input,size_output)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=args.optim_lr,
        eps=args.optim_eps,
    )

    next_obs,_ = envs.reset()
    next_mask = envs.action_masks()
    next_done = torch.zeros(size=(args.num_envs,))

    heavy_args = HeavyArgs(
            seed=generator,
            envs=envs,
            agent=agent,
            optimizer=optimizer,
            next_mask=next_mask,
            next_obs=next_obs,
            next_done=next_done,
            logger=writer,
            logger_step=0
        )

    score = torch.zeros(size=(args.num_ppo_iter,))
    for i in range(args.num_ppo_iter):
        observations, done, info = run_one_ppo_iter(args=args,heavy_args=heavy_args)
        heavy_args.next_done = done
        heavy_args.next_obs = observations
        score[i] = info["agents_scores"]
        # This counter allow to compare different runs with different num_step or num_envs
        heavy_args.logger_step += args.num_step*args.num_envs

    torch.save(agent.state_dict(),"models/agent.pt")

    writer.add_hparams(hparam_dict=dataclasses.asdict(args),metric_dict={"hparam/score":score},run_name="yams_rl1")


if __name__== "__main__":
    args = tyro.cli(Args)
    train_agent(args)
