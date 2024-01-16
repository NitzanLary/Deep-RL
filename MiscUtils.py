from collections import defaultdict
from typing import Optional, List, Tuple

import gymnasium as gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from constants import env_args, MAX_EPISODE_STEPS


def get_env(render_mode: Optional[str] = None) -> gym.Env:
    """Returns a gym environment."""
    env_args["render_mode"] = render_mode
    base_env = gym.make(**env_args)
    time_limit_env = gym.wrappers.TimeLimit(base_env, max_episode_steps=MAX_EPISODE_STEPS)
    record_episode_statistics_env = gym.wrappers.RecordEpisodeStatistics(time_limit_env, deque_size=100_000)
    return record_episode_statistics_env


def get_state_dim(env: gym.Env) -> int:
    """Returns the dimension of the state space."""
    return env.observation_space.n


def q_learning_heatmap(q_table: defaultdict, shape: Tuple[int, int]):
    q_values = np.zeros(shape)
    for key, value in q_table.items():
        row = key // shape[0]
        col = key % shape[1]
        q_values[row, col] = np.max(value)
    sns.heatmap(q_values, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


def plot_rewards_per_episode(rewards: List[float]) -> None:
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


def plot_durations(durations: List[float]) -> None:
    """Plot the durations of each episode, averaged over 100 episodes."""
    averaged_durations = [np.mean(durations[i - 100: i]) for i in range(100, len(durations), 100)]
    plt.plot(averaged_durations)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.show()


def plot_training_error(training_error: List[float]) -> None:
    plt.plot(training_error)
    plt.xlabel("Episode")
    plt.ylabel("Training Error")
    plt.show()
