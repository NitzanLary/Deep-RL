from typing import Tuple

from tqdm import tqdm

from MiscUtils import get_env


class Agent:
    def __init__(
            self,
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
    ):
        """
        :param env: The environment to train the agent on
        :param learning_rate: The learning rate
        :param initial_epsilon: The initial epsilon value
        :param epsilon_decay: The decay for epsilon
        :param final_epsilon: The final epsilon value
        :param discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []
        self.statistics = {"rewards_per_episode": [], "steps_per_episode": []}

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """

    def update(
            self,
            obs: Tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

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