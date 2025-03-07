# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

import torch

from typing import Any, Literal


LocalAction = Literal[0, 1, 2]
GameState = dict[str, Any]
Player = Literal[1, -1]
Pos = tuple[int, int]
GlobalAction = Literal["up", "right", "down", "left"]
SnakeId = str
Move = dict[Literal["move"], GlobalAction]


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

LOCAL_ACTIONS = (0, 1, 2)
LOCAL_ACTIONS_NAMES = ("left", "forward", "right")
N_LOCAL_ACTIONS = 3
GLOBAL_ACTIONS = ("up", "right", "down", "left")
N_GLOBAL_ACTIONS = 4

N_SNAKES = 2
TERRAIN_X_SIZE = 11
TERRAIN_Y_SIZE = 11
MAX_HEALTH = 100

TERRAIN_DIMS = 4
VIEW_SIZE = 3  # Careful this value is duplicated in Game/game.h
CONV_SHAPE = (N_SNAKES, 1 + 2 * VIEW_SIZE, 1 + 2 * VIEW_SIZE)
AID_SIZE = 4  # Careful this value is duplicated in Game/game.h
"""0: length | 1: left_distance_to_nearest_apple | 2: front_distance_to_nearest_apple | 3: right_distance_to_nearest_apple"""
AID_SHAPE = (N_SNAKES, AID_SIZE)

UCT_TIME = 0.1

N_RUN = 1
N_GAMES_IN_BROWSER = 1

MOVING_AVERAGE_WINDOW = 10

MAIN_PLAYER = 1
OTHER_PLAYER = -1
NO_WINNER = 0

NO_WINNER_PENALTY = -0.2
"""Value in [-1; 0] to give to a game where there is no winner. 0 means a draw, -1 means a loss. (Helps the snakes avoid draws)"""

INF = float("inf")
