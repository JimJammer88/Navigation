import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,state_size, action_size, seed, fc1_units=64,fc2_units=64,vfc1_units = 24,vfc2_units=24,
                 afc1_units=24, afc2_units = 24):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first joint hidden layer
            fc2_units (int): Number of nodes in second joint hidden layer
            vfc1_units (int): Number of nodes in the first layer of the state value function stream
            vfc2_units (int): Number of nodes in the second layer of the state value function stream
            afc1_units (int): Number of nodes in the first layer of the advantage  function stream
            afc2_units (int): Number of nodes in the second layer of the advantage function stream
            
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.vfc1 = nn.Linear(fc1_units, vfc1_units)
        self.vfc2 = nn.Linear(vfc1_units, 1)
        self.afc1 = nn.Linear(fc1_units, afc1_units)
        self.afc2 = nn.Linear(afc1_units, action_size)
        

    def forward(self, state):
        """Build a dueling network that maps state -> action values"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = F.relu(self.vfc1(x))
        v = self.vfc2(v)
        a = F.relu(self.afc1(x))
        a = self.afc2(a)
        q = v + a - torch.mean(a)
        return q
