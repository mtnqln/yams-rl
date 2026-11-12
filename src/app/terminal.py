import tyro
from dataclasses import dataclass

@dataclass
class PlayArgs:
    players: list[str]=[]
    """List of the players"""
    stochastic_agent: bool = False
    """Whether to use or not stochastic policy for the agent"""

def play(args:PlayArgs):
    ...


if __name__=="__main__":
    args = tyro.cli(PlayArgs)
    play(args)
