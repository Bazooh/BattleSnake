import numpy as np

from typing import Any

from Constants import GLOBAL_ACTIONS
from Game.cgame import DOWN, LEFT, RIGHT, UP

DIRECTIONS = {"up": np.array([0, 1]), "down": np.array([0, -1]), "right": np.array([1, 0]), "left": np.array([-1, 0])}
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


def get_grid(snakes: dict[str, Any], bounds: np.ndarray) -> np.ndarray:
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


def get_global_direction_idx(snake: dict[str, Any]) -> int:
    head = snake["body"][0]
    neck = snake["body"][1]

    direction = (head["x"] - neck["x"], head["y"] - neck["y"])

    match direction:
        case (0, 1):
            return UP
        case (0, -1):
            return DOWN
        case (1, 0):
            return RIGHT
        case (-1, 0):
            return LEFT

    assert False, f"Invalid direction {direction}"


def get_global_direction(snake: dict[str, Any]) -> str | None:
    idx = get_global_direction_idx(snake)
    return GLOBAL_ACTIONS[idx] if idx is not None else None


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
