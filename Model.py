import torch
import torch.nn as nn
import torch.nn.functional as f


class Model(nn.Module):

    def __init__(self, in_size, out_size, seed, hidden1=64, hidden2=64):
        super(Model, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_size)

    def forward(self, _input):
        x = f.relu(self.fc1(_input))
        x = f.relu(self.fc2(x))
        return self.fc3(x)
