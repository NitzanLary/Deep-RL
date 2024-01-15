from MiscUtils import get_env
from constants import LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR


class QNetwork(torch.nn.Module):
    """QNetwork class that inherits from torch.nn.Module.
    Implements a neural network with 3 fully connected layers.
    """



class BasicDQN_Agent:
    def __init__(
            self,
            env=get_env(render_mode=None),
            learning_rate=LEARNING_RATE,
            initial_epsilon=START_EPSILON,
            epsilon_decay=EPSILON_DECAY,
            final_epsilon=FINAL_EPSILON,
            discount_factor=DISCOUNT_FACTOR,
    ):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.statistics = {"rewards_per_episode": [], "steps_per_episode": []}

        self.q_network = QNetwork()
        self.target_q_network = QNetwork(self.config)
        self.replay_buffer = ReplayBuffer(self.config)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def step(self, state):
        """
        :param state: (np.ndarray) current state
        :return: (np.ndarray) action
        """
