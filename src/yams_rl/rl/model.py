import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer:nn.Linear,scl:float=np.sqrt(2),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight,scl)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self,size_input:int,size_output:int)->None:
        super().__init__()

        self.inf_mask = torch.full(fill_value=torch.finfo(torch.float32).min,size=(size_output,))

        # Network that will learn the policy
        self.actor = nn.Sequential(
            layer_init(layer=nn.Linear(size_input,128)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(128,256)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(256,512)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(512,256)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(256,128)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(128,size_output),scl=0.01)
        )
        # Network that will learn the Value function, takes the state s_t as input
        self.critic = nn.Sequential(
            layer_init(layer=nn.Linear(size_input,128)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(128,256)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(256,512)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(512,256)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(256,128)),
            nn.Tanh(),
            layer_init(layer=nn.Linear(128,1),scl=1)
        )

    def forward(self,obs):  # ty: ignore[invalid-mutable-override]
        logits = self.actor(obs)
        return logits

    def get_action_and_value(self,obs,action:torch.Tensor|None=None,mask:torch.Tensor|None=None):
        """
        Returns action,log prob, entropy and Value for a given observation and/or action
        Apply a -inf mask on the logits where a True in the mask is considered as a valid action
        """
        logits = self.actor(obs)
        if mask is not None:
            logits = logits + (~mask)*self.inf_mask
        prob = Categorical(logits=logits)
        if action is None:
            action = prob.sample()
        return action,prob.log_prob(action),prob.entropy(),self.critic(obs).squeeze(-1)

    def get_value(self,obs):
        return self.critic(obs).squeeze(-1) # to remove the last dim which is always equal to 1
