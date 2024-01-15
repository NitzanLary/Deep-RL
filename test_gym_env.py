from MiscUtils import q_learning_heatmap, plot_rewards_per_episode, plot_durations
from QLearningAgent import QLearningAgent


def assignment_1():
    """
    Implements the Q-learning algorithm.
    Train the agent for 5000 with max_episode_steps=100.
    Save agent q_values after 500 episodes, 2000 episodes, 5000 episodes.
    - plot the heatmap of the Q-values for each of the three saved q_values.
    - plot the rewards per episode.
    - plot the steps per episode.
    """
    agent = QLearningAgent()
    agent.train(500)
    agent.save_q_values("pickles/q_values_500.npy")
    agent.train(1500)
    agent.save_q_values("pickles/q_values_2000.npy")
    agent.train(3000)
    agent.save_q_values("pickles/q_values_5000.npy")

    # plot the heatmap of the Q-values for each of the three saved q_values
    for i in [500, 2000, 5000]:
        q_table = agent.load_q_values(f"pickles/q_values_{i}.npy")
        q_learning_heatmap(q_table, (4, 4))

    # plot the rewards per episode
    plot_rewards_per_episode(agent.statistics["rewards_per_episode"])

    # plot the steps per episode
    plot_durations(agent.statistics["steps_per_episode"])


if __name__ == "__main__":
    assignment_1()
