import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="critic",
        chkpt_dir="tmp/td3",
        learning_rate=1e-3
    ):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_td3")

        input_dim = input_dims if isinstance(input_dims, int) else input_dims[0]
        self.fc1 = nn.Linear(input_dim + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        print(f"{self.name} device is {self.device}")

        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q1 = self.q1(x)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print(f"Checkpoint saved to {self.checkpoint_file}")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print(f"Checkpoint loaded from {self.checkpoint_file}")

class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="actor",
        chkpt_dir="tmp/td3",
        learning_rate=1e-3
    ):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_td3")

        input_dim = input_dims if isinstance(input_dims, int) else input_dims[0]
        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        print(f"{self.name} device is {self.device}")

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = T.tanh(self.mu(x))
        return mu

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print(f"Checkpoint saved to {self.checkpoint_file}")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print(f"Checkpoint loaded from {self.checkpoint_file}")