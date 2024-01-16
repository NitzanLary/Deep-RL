env_args = {
    "id": "FrozenLake-v1",
    "desc": None,
    "map_name": "4x4",
    "is_slippery": False,
}
MAX_EPISODE_STEPS = 100
LEARNING_RATE = 0.01
N_EPISODES = 15000
START_EPSILON = 1.0
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 0.95

BATCH_SIZE = 4
