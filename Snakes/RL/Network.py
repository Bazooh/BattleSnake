import torch
import torch.nn as nn
import torch.nn.functional as F
from Snakes.RL.Utils import *

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        
        self.layer1 = nn.Linear(804, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor):
        terrain, other = x[:, :-4], x[:, -4:]
        x = terrain.unflatten(1, (5, 11, 11))
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = torch.cat((x.flatten(1), other), dim=1)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return torch.sigmoid(self.layer3(x))