# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
import torch

BATCH_SIZE = 512
EPOCHS = 100
GAMMA = 0.99
EPS_START = -1
EPS_END = -1
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 100_000
DEVICE = torch.device("cpu")

ACTIONS = ["up", "right", "down", "left"]
TERRAIN_DIMS = 4

N_ACTIONS = len(ACTIONS)
N_SNAKES = 2
TERRAIN_X_SIZE = 11
TERRAIN_Y_SIZE = 11
MAX_HEALTH = 100

UCT_TIME = 0.1

N_RUN = 200
N_GAMES_IN_BROWSER = 200

MOVING_AVERAGE_WINDOW = 10

MAIN_PLAYER = 0
OTHER_PLAYER = 1
NO_WINNER = 0.5

INF = float('inf')