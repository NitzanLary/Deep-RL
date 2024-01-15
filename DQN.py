from collections import defaultdict
import numpy as np

x = defaultdict(lambda: np.zeros(4))
x[0][0] = 1
x[3][2] = 2
# convert to dict
x = dict(x)
# save dict
np.save("q_values.npy", x)
# load dict
y = np.load("q_values.npy", allow_pickle=True).item()
y = defaultdict(lambda: np.zeros(4), y)
print((y))