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
     std=self.log_std.exp().clamp(min=1e-3)
     return torch.distributions.Normal(mean,std)
  def get_action(self,obs):
     dist=self.get_dist(obs)
     u=dist.sample()
     action=torch.tanh(u)
     log_prob=dist.log_prob(u).sum(-1)-torch.log(1-action**2+1e-6).sum(-1)#+1e-6防止log0出现梯度爆炸
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
