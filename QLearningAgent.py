from collections import defaultdict
from typing import Tuple

import numpy as np
from tqdm import tqdm

from MiscUtils import get_env
from constants import LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR


class QLearningAgent:
    def __init__(
            self,
            env=get_env(render_mode=None),
            learning_rate=LEARNING_RATE,
            initial_epsilon=START_EPSILON,
            epsilon_decay=EPSILON_DECAY,
            final_epsilon=FINAL_EPSILON,
            discount_factor=DISCOUNT_FACTOR,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The environment to train the agent on
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        # initialize the Q-value table with a default value of zero
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.statistics = {"rewards_per_episode": [], "steps_per_episode": []}

    def initialize_q_values(self, q_values: defaultdict) -> None:
        self.q_values = q_values

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
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: Tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        target = reward
        if not terminated:
            target += np.max(self.q_values[next_obs])

        new_q_value = (1 - self.lr) * self.q_values[obs][action] + self.lr * target

        self.training_error.append(abs(new_q_value - self.q_values[obs][action]))
        self.q_values[obs][action] = new_q_value

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_values(self, path: str) -> None:
        np.save(path, dict(self.q_values))

    def load_q_values(self, path: str) -> defaultdict:
        return defaultdict(lambda: np.zeros(self.env.action_space.n), np.load(path, allow_pickle=True).item())

    def train(self, n_episodes: int) -> None:
        """Train the agent for n_episodes"""
        rewards_per_episode = self.statistics["rewards_per_episode"]
        steps_per_episode = self.statistics["steps_per_episode"]
        for episode in tqdm(range(n_episodes)):
            rewards_per_episode.append(0)
            steps_per_episode.append(0)
            obs, info = self.env.reset()
            done = False

            # play one episode
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # update the agent
                self.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

                # update the episode statistics
                rewards_per_episode[-1] += reward
                steps_per_episode[-1] += 1
                if truncated:
                    steps_per_episode[-1] = 100

            self.decay_epsilon()

        self.env.close()

    def test_agent(self, n_episodes: int) -> None:
        """Test the agent for n_episodes"""
        test_env = get_env(render_mode="human")
        for episode in range(n_episodes):
            obs, info = test_env.reset()
            done = False

            # play one episode
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = test_env.step(action)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

                test_env.render()

        test_env.close()
