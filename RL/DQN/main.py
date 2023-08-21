from random import choice, random, sample
from time import sleep
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.optim as optim
from collections import deque
from typing import NamedTuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)
INPUT_SHAPE = 2 # x & y coordinates
ACTION_SPACE = 4 # Right, Left, Up, Down
MAX_H = 20 # Max steps in episode
BATCH_SIZE = 64
TRAIN_EACH = 4 # Train each step

MAP = np.array([
    [0 , 0 , 0 ,-1 , 5],
    [0 , 0 ,0,-1 , 0],
    [1 , 0 , 5 , 0 , 0],
    [-2 , 0 ,0 ,-1, 0],
    [2 , 0 , 0 , 0, 0]
], dtype=float)

LR = 5e-4
discount = 0.99

Transition = NamedTuple("Transition", [('state', np.ndarray), ('action', int), ('reward', float), ('next_state', np.ndarray), ('terminal', bool)])

# Takes state, returns Q values for each action
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 6)
        self.fc3 = nn.Linear(6, 8)
        self.fc4 = nn.Linear(8, 6)

        self.fc5 = nn.Linear(6, output_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class Memory(deque):
    def __init__(self):
        super().__init__([], maxlen=BUFFER_SIZE)

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
        experiences: list[Transition] = sample(self, k=min(len(self) - 1,BATCH_SIZE))
        
        states_tensor = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions_tensor = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards_tensor = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states_tensor = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        terminals_tensor = torch.from_numpy(np.vstack([e.terminal for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, terminals_tensor
    
class Agent():
    def __init__(self, state_size: int, action_size: int) -> None:
        self.state_size = state_size
        self.action_size = action_size

        self.Q_o = QNetwork(state_size, action_size).to(device) # Online network
        self.Q_t = QNetwork(state_size, action_size).to(device) # Target network

        self.Q_t.load_state_dict(self.Q_o.state_dict())

        # Define optimizer and criterion
        self.optimizer = optim.SGD(self.Q_o.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()

        self.memory = Memory()

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
            action = choice(range(self.action_size))

        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        self.memory.append(Transition(state, action, reward, next_state, terminal))

        self.s_since_train = (self.s_since_train + 1) % TRAIN_EACH
        if self.s_since_train == 0:
            self.train()

    def train(self):
        states, actions, rewards, next_states, terminals = self.memory.batch()

        self.Q_o.train()
        self.Q_t.eval()

        predicted = self.Q_o(states).gather(1, actions) # BATCH_SIZE of Tensor<ExpectedRewardOfTakenAction>
        labels_next = self.Q_t(next_states).detach().max(1)[0].unsqueeze(1) # BATCH_SIZE of Tensor<ExpectedRewardOfNextActionUsingOfflineNetwork>

        targets = rewards + ( discount * labels_next * (1 - terminals) )
        loss = self.criterion(predicted, targets).to(device)
        
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

    def show_policy(self):
        actions_str = ""
        for y in range(MAP.shape[0]):
            line = ""
            for x in range(MAP.shape[1]):
                out = self.Q_o(torch.tensor((x, y), dtype=torch.float32).to(device)).cpu().detach().numpy()
                action = np.argmax(out)
                if action == 0: # Right
                    line += "→"
                if action == 1: # Left
                    line += "←"
                if action == 2: # Up
                    line += "↑"
                if action == 3: # Down
                    line += "↓"

            actions_str += line + "\n"

        print(actions_str)

def episode(g):
    Map = MAP.copy()
    d = []
    x, y = 0, 0 # Agent coords
    for s in range(MAX_H):
        state = np.array([x, y])
        # Choose action
        
        action = agent.act(state, g)
        # action = int(input(f"TFK ? {state}"))

        # Execute action
        new_x, new_y = x, y
        if action == 0: # Right
            new_x = np.clip(x + 1, 0, MAP.shape[1]-1)
        if action == 1: # Left
            new_x = np.clip(x - 1, 0, MAP.shape[1]-1)
        if action == 2: # Up
            new_y = np.clip(y - 1, 0, MAP.shape[0]-1)
        if action == 3: # Down
            new_y = np.clip(y + 1, 0, MAP.shape[0]-1)

        # Get reward
        reward = Map[new_y, new_x]

        # print(f"Reward is {reward}")

        if reward > 0:
            Map[new_y, new_x] = 0.0

        terminal = reward < 0

        # if terminal: print("C fini")

        # Store in history
        d.append(Transition(np.array((x, y)), action, reward, np.array((x, y)), terminal))
        agent.step(np.array((x, y)), action, reward, np.array((x, y)), terminal)

        x, y = new_x, new_y

        if terminal:
            # Terminal state
            break

    return d

agent = Agent(INPUT_SHAPE, ACTION_SPACE)

print(MAP)

def play_episode(e: list[Transition]):
    for i in range(len(e)):
        s = e[i]

        print(f"Frame {i}")
        print("-----------------------")
        

        state = s.state
        map_str = ""
        for row in range(MAP.shape[0]):
            line = ""
            for col in range(MAP.shape[1]):
                cell = MAP[row, col]
                if state[0] == col and state[1] == row:
                    line += "P"
                elif cell < 0:
                    line += "X"
                elif cell > 0:
                    line += "Y"
                else:
                    line += " "
            map_str += line + "\n"
        
        print(map_str)
        print("-----------------------")
        sleep(1)

TRAINING_STEPS = 100000
for i in range(TRAINING_STEPS):
    ratio = i / TRAINING_STEPS
    if ratio < 0.1:
        greed = 0.0
    elif ratio < 0.2:
        greed = 0.1
    elif ratio < 0.4:
        greed = 0.3
    elif ratio < 0.5:
        greed = 0.7
    else:
        greed = 0.95
    episode(greed) # Greed increases over time

    if i % 100 == 0:
        print(f"Epoch {int(i / 100)} / {int(TRAINING_STEPS/100)} ; Greed = {greed}")
        #play_episode(episode(i / TRAINING_STEPS))

d = episode(1.0)
play_episode(d)
print(d)
agent.show_policy()