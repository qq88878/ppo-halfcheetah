import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
  def __init__(self,state_dim=17,action_dim=6,hidden=64):
     super().__init__()
     self.fc1=nn.Linear(state_dim,hidden)
     self.fc2=nn.Linear(hidden,hidden)
     self.fc3=nn.Linear(hidden,action_dim)
     self.log_std=nn.Parameter(torch.zeros(6)-0.5)
  def forward(self,obs):
     x=F.relu(self.fc1(obs))
     x=F.relu(self.fc2(x))
     return self.fc3(x)
  def get_dist(self,obs):
     mean=self.forward(obs)
     std=self.log_std.exp()
     return torch.distributions.Normal(mean,std)
  def get_action(self,obs):
     dist=self.get_dist(obs)
     action=dist.sample()
     log_prob=dist.log_prob(action).sum(-1)
     return action,log_prob
class Critic(nn.Module):
 def __init__(self,obs=17,hidden=64):
     super().__init__()
     self.fc1=nn.Linear(obs,hidden)
     self.fc2=nn.Linear(hidden,hidden)
     self.fc3=nn.Linear(hidden,1)
 def forward(self,obs):
     x=F.relu(self.fc1(obs))
     x=F.relu(self.fc2(x))
     return self.fc3(x)
