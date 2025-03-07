from __future__ import annotations

import ctypes
import torch

from typing import Any, Literal, Sequence, Type, TypeVar, cast

from Constants import NO_WINNER, GLOBAL_ACTIONS, GlobalAction

X = 0
Y = 1

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

N_LOCAL_ACTIONS = 3
LocalAction = Literal[0, 1, 2]
Pos = tuple[int, int]
CNumber = type[ctypes.c_int] | type[ctypes.c_float]

LP_c_int = ctypes.POINTER(ctypes.c_int)
LP_LP_c_int = ctypes.POINTER(LP_c_int)
LP_c_float = ctypes.POINTER(ctypes.c_float)
LP_LP_c_float = ctypes.POINTER(LP_c_float)
LP_LP_LP_c_float = ctypes.POINTER(LP_LP_c_float)

my_library = ctypes.CDLL("/Users/aymeric/Desktop/Programming/AI/BattleSnake/Game/game.so")


T = TypeVar("T", bound=ctypes.c_int | ctypes.c_float)


def c_array(array: Sequence, type: Type[T] = ctypes.c_int) -> ctypes.Array[T]:
    return (type * len(array))(*array)


def c_matrix(matrix: list[list], type: CNumber = ctypes.c_int):
    c_arrays = (ctypes.POINTER(type) * len(matrix))()

    for i, inner_list in enumerate(matrix):
        c_arrays[i] = c_array(inner_list, type)

    return ctypes.cast(c_arrays, ctypes.POINTER(ctypes.POINTER(type)))


def c_3d_matrix(matrix_3d: list[list[list]], type: CNumber = ctypes.c_int):
    c_matrices = (ctypes.POINTER(ctypes.POINTER(type)) * len(matrix_3d))()

    for i, matrix in enumerate(matrix_3d):
        c_matrices[i] = c_matrix(matrix, type)

    return ctypes.cast(c_matrices, ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(type))))


def matrix_from_list(from_list: list[list[Pos]], width: int, height: int) -> list[list[int]]:
    matrix = []
    for _ in range(width):
        matrix.append([0 for _ in range(height)])
    for sub_l in from_list:
        for item in sub_l:
            matrix[item[0]][item[1]] = 1
    return matrix


class CSnakePart(ctypes.Structure):
    def __init__(self, x, y):
        super(CSnakePart, self).__init__()

        self.x = x
        self.y = y
        self.next: Any = None
        self.prev: Any = None


LP_c_SnakePart = ctypes.POINTER(CSnakePart)

CSnakePart._fields_ = [
    ("x", ctypes.c_int),
    ("y", ctypes.c_int),
    ("next", LP_c_SnakePart),
    ("prev", LP_c_SnakePart),
]


class CSnake(ctypes.Structure):
    _fields_ = [
        ("head", LP_c_SnakePart),
        ("tail", LP_c_SnakePart),
        ("health", ctypes.c_int),
        ("global_direction", ctypes.c_int),
        ("playable_actions", ctypes.c_bool * N_LOCAL_ACTIONS),
    ]

    def __init__(self, body: list[Pos], health: int = 100, width: int = 11, height: int = 11):
        super(CSnake, self).__init__()

        cbody = [CSnakePart(x, y) for x, y in body]
        cbody[0].next = ctypes.pointer(cbody[1])

        for i in range(1, len(cbody) - 1):
            cbody[i].next = ctypes.pointer(cbody[i + 1])
            cbody[i].prev = ctypes.pointer(cbody[i - 1])
        cbody[-1].prev = ctypes.pointer(cbody[-2])

        self.head = ctypes.pointer(cbody[0])
        self.tail = ctypes.pointer(cbody[-1])
        self.health = health
        self.global_direction = (
            UP
            if body[0][Y] > body[1][Y]
            else DOWN
            if body[0][Y] < body[1][Y]
            else RIGHT
            if body[0][X] > body[1][X]
            else LEFT
        )

        local_directions = [(self.global_direction + i) & 0b11 for i in [-1, 0, 1]]
        playable_actions = (
            body[0][X] + dx >= 0 and body[0][X] + dx < width and body[0][Y] + dy >= 0 and body[0][Y] + dy < height
            for dx, dy in [[(0, 1), (1, 0), (0, -1), (-1, 0)][direction] for direction in local_directions]
        )
        self.playable_actions = (ctypes.c_bool * N_LOCAL_ACTIONS)(*playable_actions)

    def __iter__(self):
        current = self.head.contents
        while True:
            yield current
            if not current.next:
                break
            current = current.next.contents

    def __str__(self) -> str:
        return str([(part.x, part.y) for part in self])

    def __len__(self) -> int:
        return sum(1 for _ in self)


LP_c_Snake = ctypes.POINTER(CSnake)
LP_LP_c_Snake = ctypes.POINTER(LP_c_Snake)


class Board:
    def __init__(
        self,
        width: int,
        height: int,
        snakes: list[list[Pos]],
        snakes_health: tuple[int, ...] | None = None,
        apples: list[Pos] = [],
        winner: int = NO_WINNER,
        finished: bool = False,
        turn: int = 0,
    ):
        self.width = width
        self.height = height
        self.snakes = snakes
        self.snakes_health = snakes_health if snakes_health is not None else [100 for _ in range(len(snakes))]
        self.apples = apples
        self.winner = winner
        self.finished = finished
        self.turn = turn

    @staticmethod
    def from_cboard(cboard: CBoard) -> Board:
        return Board(
            width=cast(int, cboard.width),
            height=cast(int, cboard.height),
            snakes=[[(part.x, part.y) for part in cboard.snakes[i]] for i in range(cast(int, cboard.nb_snakes))],
            snakes_health=tuple(cast(int, cboard.snakes[i].contents.health) for i in range(cast(int, cboard.nb_snakes))),
            apples=[
                (i, j)
                for i in range(cast(int, cboard.width))
                for j in range(cast(int, cboard.height))
                if cboard.apples_matrix[i][j] == 1
            ],
            winner=cast(int, cboard.winner),
            finished=cast(bool, cboard.finished),
            turn=cboard.turn,
        )

    @staticmethod
    def from_game_state(game_state: dict) -> Board:
        main_snake = [(part["x"], part["y"]) for part in game_state["you"]["body"]]
        other_snakes = [
            [(part["x"], part["y"]) for part in snake["body"]]
            for snake in game_state["board"]["snakes"]
            if snake["id"] != game_state["you"]["id"]
        ]
        return Board(
            width=game_state["board"]["width"],
            height=game_state["board"]["height"],
            snakes=[main_snake] + other_snakes,
            snakes_health=(game_state["you"]["health"],)
            + tuple(snake["health"] for snake in game_state["board"]["snakes"] if snake["id"] != game_state["you"]["id"]),
            apples=[(apple["x"], apple["y"]) for apple in game_state["board"]["food"]],
            turn=game_state["turn"],
        )

    def __str__(self) -> str:
        return str(CBoard.from_board(self))


class CBoard(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("nb_snakes", ctypes.c_int),
        ("snakes_matrix", LP_LP_c_int),
        ("apples_matrix", LP_LP_c_int),
        ("snakes", LP_LP_c_Snake),
        ("winner", ctypes.c_int),
        ("finished", ctypes.c_bool),
        ("turn", ctypes.c_int),
        ("convs", LP_LP_LP_c_float),
        ("aids", LP_LP_c_float),
    ]

    _free_board = my_library.free_board

    def __init__(
        self,
        width,
        height,
        nb_snakes,
        snakes_matrix,
        apples_matrix,
        snakes,
        convs,
        aids,
        winner: int = NO_WINNER,
        finished: bool = False,
        turn: int = 0,
    ):
        super(CBoard, self).__init__()

        self.width: ctypes.c_int = width
        self.height: ctypes.c_int = height
        self.nb_snakes: ctypes.c_int = nb_snakes
        self.snakes_matrix = snakes_matrix
        self.apples_matrix = apples_matrix
        self.snakes = snakes
        self.convs = convs
        self.aids = aids
        self.winner: ctypes.c_int = cast(ctypes.c_int, winner)
        self.finished: ctypes.c_bool = cast(ctypes.c_bool, finished)
        self.turn = turn
        self.should_be_freed = False

    @classmethod
    def from_board(cls, board: Board):
        return cls(
            width=board.width,
            height=board.height,
            nb_snakes=len(board.snakes),
            snakes_matrix=c_matrix(matrix_from_list(board.snakes, board.width, board.height)),
            apples_matrix=c_matrix(matrix_from_list([board.apples], board.width, board.height)),
            snakes=(LP_c_Snake * len(board.snakes))(
                *(
                    ctypes.pointer(CSnake(snake, health, board.width, board.height))
                    for snake, health in zip(board.snakes, board.snakes_health)
                )
            ),
            convs=c_3d_matrix(cast(list[list[list]], torch.zeros((2, 7, 7))), ctypes.c_float),
            aids=c_matrix(cast(list[list], torch.zeros((2, 4))), ctypes.c_float),
            winner=board.winner,
            finished=board.finished,
            turn=board.turn,
        )

    @classmethod
    def from_game_state(cls, game_state: dict):
        return cls.from_board(Board.from_game_state(game_state))

    def next(self, local_directions: tuple[LocalAction, LocalAction]) -> CBoard:
        """Return the next board after the snakes move in the given directions

        Args:
            directions (Tuple[int, int]): Tuple of two integers representing the local direction of the snakes
                0: left | 1: straight | 2: right

        Returns:
            CBoard: The next board
        """

        next_board = _next_board(ctypes.byref(self), c_array(local_directions)).contents
        next_board.should_be_freed = True

        return next_board

    def __str__(self) -> str:
        board_str = ""
        for y in range(cast(int, self.height) - 1, -1, -1):
            for x in range(cast(int, self.width)):
                for snake_idx in range(cast(int, self.nb_snakes)):
                    snake = self.snakes[snake_idx].contents
                    if x == snake.head.contents.x and y == snake.head.contents.y:
                        if snake.global_direction == UP:
                            board_str += "^ "
                        elif snake.global_direction == RIGHT:
                            board_str += "> "
                        elif snake.global_direction == DOWN:
                            board_str += "v "
                        else:
                            board_str += "< "
                        break
                else:
                    if self.apples_matrix[x][y] == 1:
                        board_str += "O "
                    elif self.snakes_matrix[x][y] == 1:
                        board_str += "* "
                    else:
                        board_str += ". "
            board_str += "\n"
        return board_str

    def to_tensors(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of three tensors:
            - (N_SNAKES, *CONV_SHAPE[1:]) Conv tensor representing the view of the snakes
            - (N_SNAKES, *AID_SHAPE[1:]) Tensor representing some helpful parameters for the snake (length, left_dist, front_dist, right_dist)
            - (N_SNAKES, 3) Tensor representing the possible actions for the snake
        """
        if self.finished:
            raise ValueError("Cannot convert a finished board to tensors")

        return (
            torch.tensor(
                [[[self.convs[i][j][k] for k in range(7)] for j in range(7)] for i in range(cast(int, self.nb_snakes))],
                dtype=torch.float,
                device=device,
            ),
            torch.tensor(
                [[self.aids[i][j] for j in range(4)] for i in range(cast(int, self.nb_snakes))],
                dtype=torch.float,
                device=device,
            ),
            torch.tensor([snake.contents.playable_actions for snake in self.snakes[:2]], dtype=torch.bool, device=device),
        )

    def has_same_snakes(self, other: CBoard) -> bool:
        return (
            self.nb_snakes == other.nb_snakes
            and all(
                len(self.snakes[snake_idx].contents) == len(other.snakes[snake_idx].contents)
                and all(
                    part.x == other_part.x and part.y == other_part.y
                    for part, other_part in zip(self.snakes[snake_idx].contents, other.snakes[snake_idx].contents)
                )
                for snake_idx in range(cast(int, self.nb_snakes))
            )
            and all(
                self.snakes[snake_idx].contents.health == other.snakes[snake_idx].contents.health
                for snake_idx in range(cast(int, self.nb_snakes))
            )
        )

    def local_action_to_global(self, local_action: LocalAction, snake_idx: int) -> GlobalAction:
        action = (self.snakes[snake_idx].contents.global_direction + local_action - 1) & 0b11  # % 4
        return GLOBAL_ACTIONS[action]

    def global_action_to_local(self, global_action: int, snake_idx: int) -> LocalAction:
        action = (global_action - self.snakes[snake_idx].contents.global_direction + 1) & 0b11
        return cast(LocalAction, action)

    def __del__(self):
        if self.should_be_freed:
            self._free_board(ctypes.pointer(self))

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, CBoard), f"Cannot compare CBoard with {type(other)}"

        return (
            self.snakes_matrix == other.snakes_matrix
            and self.apples_matrix == other.apples_matrix
            and self.snakes == other.snakes
            and self.winner == other.winner
            and self.finished == other.finished
            and self.turn == other.turn
        )


LP_c_Board = ctypes.POINTER(CBoard)
CBoard._free_board.argtypes = [LP_c_Board]

_next_board = my_library.next_board
_next_board.argtypes = [LP_c_Board, LP_c_int]
_next_board.restype = LP_c_Board
