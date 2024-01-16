from BasicDQN_Agent import BasicDQN_Agent
from MiscUtils import q_learning_heatmap, plot_rewards_per_episode, plot_durations, get_state_dim, plot_training_error
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


def assignment_1_deep():
    agent = BasicDQN_Agent()
    # agent.load_weights("pickles/q_network_weights_5.pt")
    agent.train(3500)
    agent.save_weights("pickles/q_network_weights_5.pt")

    # plot training error
    plot_training_error(agent.training_error)

    agent.test_agent(2)


if __name__ == "__main__":
    # assignment_1()
    assignment_1_deep()
