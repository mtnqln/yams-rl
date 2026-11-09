import torch
from torch.distributions import Categorical

from yams_rl.rl.encoding import build_action_mask, build_observation
from yams_rl.rl.model import Agent
from yams_rl.rl.utils import nn_output_to_action
from yams_rl.yams.categories import Action
from yams_rl.yams.game_state import GameState
from yams_rl.yams.players import Player


class AgentPlayer(Player):

    def __init__(self,model:Agent,agent_id:int,stochastic:bool=True):
        super().__init__()

        self.name = str(agent_id)
        self.agent_id = agent_id
        self.stochastic = stochastic
        self.model = model
        self.inf_mask = torch.full(fill_value=torch.finfo(torch.float32).min,size=(self.model.size_output,))

    def act(self, state: GameState) -> Action:
        with torch.no_grad():
            assert state.current_player_id==self.agent_id, "Desynchronisation between agent id and the current player id"
            obs = torch.from_numpy(build_observation(state=state,nb_players=state.players_number,agent_id=self.agent_id))
            mask = torch.from_numpy(build_action_mask(state=state,dice_number=state.dices.dice_number))
            logits = self.model(obs)
            logits = logits + (~mask)*self.inf_mask
            if self.stochastic:
                prob = Categorical(logits=logits)
                nn_output = prob.sample()
                return nn_output_to_action(int(nn_output.item()))
            else:
                nn_output = torch.argmax(logits)
                return nn_output_to_action(int(nn_output.item()))
