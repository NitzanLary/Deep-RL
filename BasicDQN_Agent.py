from typing import Dict, Tuple

import numpy as np
import torch

from Agent import Agent
from MiscUtils import get_env
from constants import LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR, BATCH_SIZE


class QNetwork(torch.nn.Module):
    """QNetwork class that inherits from torch.nn.Module.
    Implements a neural network with 3 fully connected layers.
    """

    def __init__(self, config: Dict[str, int]):
        super(QNetwork, self).__init__()
        self.config = config
        self.fc1 = torch.nn.Linear(self.config["state_dim"], 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc2_1 = torch.nn.Linear(64, 64)
        self.fc2_2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, self.config["action_dim"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        :param states: A batch of states
        :return: A batch of Q-values
        """
        # assert states.shape == (self.config["batch_size"], self.config["state_dim"])
        # assert states.shape == (32, 1), f"states.shape: {states.shape} != {(32, 1)}"
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc2_2(x))
        return self.fc3(x)

    def update(self, y, targets):
        """
        update the Q-network
        :param y: The Q-values of the actions that were taken
        :param targets: A batch of targets
        :return: the loss (as a float)
        """
        self.optimizer.zero_grad()
        loss = self.loss_fn(y, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class BasicDQN_Agent(Agent):
    def __init__(
            self,
            env=get_env(render_mode=None),
            learning_rate=LEARNING_RATE,
            initial_epsilon=START_EPSILON,
            epsilon_decay=EPSILON_DECAY,
            final_epsilon=FINAL_EPSILON,
            discount_factor=DISCOUNT_FACTOR,
            batch_size=BATCH_SIZE,
            c=5,
    ):
        """Initialize super class and initialize the Q-network, target Q-network"""
        super(BasicDQN_Agent, self).__init__(
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
        )
        self.config = {
            "state_dim": 1,
            "action_dim": self.env.action_space.n,
        }
        self.batch_size = batch_size
        self.c = c  # update target network every c steps
        self.q_network = QNetwork(self.config)
        self.target_q_network = QNetwork(self.config)

        self.reply_buffer = np.empty((0, 5), dtype=object)

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.q_network.device).unsqueeze(0)
            action = torch.argmax(self.q_network.forward(obs)).item()
            return action

    def save_weights(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(torch.load(path))

    def update(
            self,
            obs: int,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: int,
    ):
        """Updates the Q-value of an action."""
        # updating the replay buffer
        triplet = np.array([[obs, action, reward, terminated, next_obs]])
        self.reply_buffer = np.append(self.reply_buffer, triplet, axis=0)
        if len(self.reply_buffer) >= self.batch_size:
            self.train_on_batch()
        if self.env.episode_count % self.c == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train_on_batch(self):
        """Train the agent on a batch of transitions."""
        batch = self.reply_buffer[np.random.choice(len(self.reply_buffer), self.batch_size, replace=False)]
        states = torch.tensor([x[0] for x in batch], dtype=torch.float32, device=self.q_network.device).unsqueeze(1)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.q_network.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.q_network.device)
        terminated = torch.tensor([x[3] for x in batch], dtype=torch.bool, device=self.q_network.device)
        next_states = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.q_network.device).unsqueeze(1)

        next_q_values = self.target_q_network.forward(next_states)
        target_q_values = rewards + self.discount_factor * torch.max(next_q_values, dim=1)[0] * ~terminated

        q_values = self.q_network.forward(states)
        # take the q_values of the actions that were taken
        y = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.q_network.update(y, target_q_values)

        self.training_error.append(loss)
