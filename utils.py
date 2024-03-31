import typing
import numpy as np
from copy import deepcopy
from itertools import product

DIRECTIONS = {
    "up": np.array([0, 1]),
    "down": np.array([0, -1]),
    "right": np.array([1, 0]),
    "left": np.array([-1, 0])
}
DEFAULT_MOVE = "down"

BLANK_CASE = 0
SNAKE_CASE = 1

SNAKE_HEAD = 0
SNAKE_TAIL = -1

BODY = 0
ID = 1

MAIN_PLAYER = 1
OTHER_PLAYER = -1

X = 0
Y = 1

def get_grid(snakes: typing.Dict, bounds: np.ndarray) -> np.ndarray:
    grid = np.zeros((bounds[X], bounds[Y]))
    
    for snake in snakes.values():
        for case in snake[1:]:
            grid[case[X], case[Y]] = SNAKE_CASE

    return grid


def collides_with_other_snake(grid: np.ndarray, head: np.ndarray) -> bool:
    return grid[head[X], head[Y]] == SNAKE_CASE


def is_in_grid(grid: np.ndarray, pos: np.ndarray) -> bool:
    return 0 <= pos[X] < grid.shape[X] and 0 <= pos[Y] < grid.shape[Y]


def is_move_valid(grid: np.ndarray, new_head: np.ndarray) -> bool:
    return is_in_grid(grid, new_head) and not collides_with_other_snake(grid, new_head)


def valid_moves(grid: np.ndarray, head: np.ndarray) -> list:
    moves: list = []
    for direction, direction_vector in DIRECTIONS.items():
        if is_move_valid(grid, head + direction_vector):
            moves.append(direction)
    return moves

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]