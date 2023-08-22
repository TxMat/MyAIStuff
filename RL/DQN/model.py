# 
# Heavily inspired by https://unnatsingh.medium.com/deep-q-network-with-pytorch-d1ca6f40bfda
#

from collections import deque
from random import sample, random, choice
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = NamedTuple("Transition", [('state', np.ndarray), ('action', int), ('reward', float), ('next_state', np.ndarray), ('terminal', bool)])

class ModelConfig():
    def __init__(
            self, 
            INPUT_SHAPE: int, 
            ACTION_SPACE: int,
            GAMMA: float = 0.99, 
            BUFFER_SIZE: int = int(1e5), 
            TRAIN_EACH: int = 4,
            BATCH_SIZE: int = 64,
            LR: float = 5e-4,
            TAU: float = 1e-3
    ):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.ACTION_SPACE = ACTION_SPACE
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.TRAIN_EACH = TRAIN_EACH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.TAU = TAU
        
# Takes state, returns Q values for each action
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 64)

        self.fc5 = nn.Linear(64, output_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class Memory(deque):
    def __init__(self, config: ModelConfig):
        self.C = config
        super().__init__([], maxlen=self.C.BUFFER_SIZE)

    def __setitem__(self, index: int, item: Transition):
        super().__setitem__(index, item)

    def __getitem__(self, index: int) -> Transition:
        return super().__getitem__(index)

    def insert(self, index: int, item: Transition):
        super().insert(index, item)

    def append(self, item: Transition):
        super().append(item)

    def extend(self, other: list[Transition]):
        super().extend(other)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ,torch.Tensor]:
        experiences: list[Transition] = sample(self, k=min(len(self) - 1,self.C.BATCH_SIZE))
        
        states_tensor = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions_tensor = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards_tensor = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states_tensor = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        terminals_tensor = torch.from_numpy(np.vstack([e.terminal for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, terminals_tensor
    
class Agent():
    def __init__(self, config: ModelConfig) -> None:
        self.Q_o = QNetwork(config.INPUT_SHAPE, config.ACTION_SPACE).to(device) # Online network
        self.Q_t = QNetwork(config.INPUT_SHAPE, config.ACTION_SPACE).to(device) # Target network

        self.Q_t.load_state_dict(self.Q_o.state_dict())

        self.optimizer = optim.Adam(self.Q_o.parameters(), lr=config.LR)

        self.memory = Memory(config)

        self.C = config

        self.s_since_train = 0
        self.l_since_reset = 0

    def act(self, state: np.ndarray, g) -> int:
        if random() < g:
            # Let agent choose
            _state = torch.from_numpy(state).float().to(device)
            self.Q_o.eval()
            with torch.no_grad():
                action = int(np.argmax(self.Q_o(_state).cpu().detach().numpy()))
            self.Q_o.train()
        else:
            # Choose random
            action = choice(range(self.C.ACTION_SPACE))

        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        self.memory.append(Transition(state, action, reward, next_state, terminal))

        self.s_since_train = (self.s_since_train + 1) % self.C.TRAIN_EACH
        if self.s_since_train == 0:
            self.train()

    def train(self):
        states, actions, rewards, next_states, terminals = self.memory.batch()

        criterion = nn.SmoothL1Loss()

        self.Q_o.train()
        self.Q_t.eval()

        predicted = self.Q_o(states).gather(1, actions) # BATCH_SIZE of Tensor<ExpectedRewardOfTakenAction>
        with torch.no_grad():
            labels_next = self.Q_t(next_states).detach().max(1)[0].unsqueeze(1) # BATCH_SIZE of Tensor<ExpectedRewardOfNextActionUsingOfflineNetwork>

        targets = rewards + ( self.C.GAMMA * labels_next * (1 - terminals) )
        loss = criterion(predicted, targets).to(device)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.l_since_reset = (self.l_since_reset + 1) % 3

        self.soft_update(self.Q_o, self.Q_t, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)