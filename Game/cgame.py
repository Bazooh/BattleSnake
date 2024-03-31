from __future__ import annotations
import ctypes
from typing import List, Tuple
import torch

NO_WINNER = 0.5

LP_c_int = ctypes.POINTER(ctypes.c_int)
LP_LP_c_int = ctypes.POINTER(LP_c_int)

my_library = ctypes.CDLL('/Users/aymeric/Desktop/AI/BattleSnake/Game/game.so')

def c_array(array: List[int]) -> LP_c_int:
    return (ctypes.c_int * len(array))(*array)

def c_matrix(matrix: List[List[int]]) -> LP_LP_c_int:
    c_arrays = (LP_c_int * len(matrix))()

    for i, inner_list in enumerate(matrix):
        c_arrays[i] = c_array(inner_list)

    return ctypes.cast(c_arrays, LP_LP_c_int)

def matrix_from_list(l: List[List[int]], width: int, height: int) -> List[List[int]] :
    matrix = []
    for _ in range(width) :
        matrix.append([0 for _ in range(height)])
    for sub_l in l :
        for item in sub_l :
            matrix[item[0]][item[1]] = 1
    return matrix

class CSnakePart(ctypes.Structure):
    def __init__(self, x, y) :
        super(CSnakePart, self).__init__()
        
        self.x = x
        self.y = y
        self.next = None
        self.prev = None

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
    ]
    
    def __init__(self, body: List[Tuple[int]], health: int = 100) :
        super(CSnake, self).__init__()
        
        cbody = [CSnakePart(x, y) for x, y in body]
        cbody[0].next = ctypes.pointer(cbody[1])
        
        for i in range(1, len(cbody) - 1) :
            cbody[i].next = ctypes.pointer(cbody[i+1])
            cbody[i].prev = ctypes.pointer(cbody[i-1])
        cbody[-1].prev = ctypes.pointer(cbody[-2])
        
        self.head = ctypes.pointer(cbody[0])
        self.tail = ctypes.pointer(cbody[-1])
        self.health = health
    
    def __iter__(self) :
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


class Board():
    def __init__(self, width: int, height: int, snakes: List[List[Tuple[int]]], apples: List[Tuple[int]], winner: int = NO_WINNER, finished: bool = False) :
        self.width = width
        self.height = height
        self.snakes = snakes
        self.apples = apples
        self.winner = winner
        self.finished = finished
    
    @staticmethod
    def from_cboard(cboard: CBoard) -> Board:
        return Board(
            cboard.width,
            cboard.height,
            [[(part.x, part.y) for part in cboard.snakes[i]] for i in range(cboard.nb_snakes)],
            [(i, j) for i in range(cboard.width) for j in range(cboard.height) if cboard.apples_matrix[i][j] == 1],
            cboard.winner,
            cboard.finished
        )
    
    @staticmethod
    def from_game_state(game_state: dict) -> Board :
        main_snake = [(part["x"], part["y"]) for part in game_state["you"]["body"]]
        other_snakes = [[(part["x"], part["y"]) for part in snake["body"]] for snake in game_state["board"]["snakes"] if snake["id"] != game_state["you"]["id"]]
        return Board(
            game_state["board"]["width"],
            game_state["board"]["height"],
            [main_snake] + other_snakes,
            [(apple["x"], apple["y"]) for apple in game_state["board"]["food"]],
        )
    
    def __str__(self) -> str:
        return str(CBoard(self))


class CBoard(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("nb_snakes", ctypes.c_int),
        ("snakes_matrix", LP_LP_c_int),
        ("apples_matrix", LP_LP_c_int),
        ("snakes", LP_LP_c_Snake),
        ("winner", ctypes.c_float),
        ("finished", ctypes.c_bool),
    ]
    
    _free_board = my_library.free_board
    
    def __init__(self, width, height, nb_snakes, snakes_matrix, apples_matrix, snakes, winner = NO_WINNER, finished = False) :
        super(CBoard, self).__init__()
        
        self.width: ctypes.c_int = width
        self.height: ctypes.c_int = height
        self.nb_snakes: ctypes.c_int = nb_snakes
        self.snakes_matrix: LP_LP_c_int = snakes_matrix
        self.apples_matrix: LP_LP_c_int = apples_matrix
        self.snakes: LP_LP_c_Snake = snakes
        self.winner: ctypes.c_float = winner
        self.finished: ctypes.c_bool = finished


    @staticmethod
    def from_board(board: Board) :
        return CBoard(
            board.width,
            board.height,
            len(board.snakes),
            c_matrix(matrix_from_list(board.snakes, board.width, board.height)),
            c_matrix(matrix_from_list([board.apples], board.width, board.height)),
            (LP_c_Snake * len(board.snakes))(*(ctypes.pointer(CSnake(snake)) for snake in board.snakes)),
            board.winner,
            board.finished
        )
    
    @staticmethod
    def from_game_state(game_state: dict) :
        return CBoard.from_board(Board.from_game_state(game_state))
    
    def __str__(self) -> str :
        board_str = ""
        for i in range(self.height - 1, -1, -1) :
            for j in range(self.width) :
                if self.apples_matrix[j][i] == 1 :
                    board_str += "O "
                elif self.snakes_matrix[j][i] == 1 :
                    board_str += "X "
                else :
                    board_str += ". "
            board_str += "\n"
        return board_str

    def to_tensor(self) -> torch.Tensor :
        terrain = torch.tensor([[self.snakes_matrix[j][i] for j in range(self.width)] for i in range(self.height)], dtype=torch.float).flatten()
        apples = torch.tensor([[self.apples_matrix[j][i] for j in range(self.width)] for i in range(self.height)], dtype=torch.float).flatten()
        main_head = torch.zeros((self.width, self.height), dtype=torch.float)
        main_head[self.snakes[0].contents.head.contents.x][self.snakes[0].contents.head.contents.y] = 1.
        other_head = torch.zeros((self.width, self.height), dtype=torch.float)
        other_head[self.snakes[1].contents.head.contents.x][self.snakes[1].contents.head.contents.y] = 1.
        
        return torch.cat(
            (
                terrain,
                apples,
                main_head.flatten(),
                other_head.flatten(),
                torch.tensor(
                    [
                        len(self.snakes[0].contents),
                        len(self.snakes[1].contents),
                        self.snakes[0].contents.health,
                        self.snakes[1].contents.health,
                    ],
                    dtype=torch.float
                ),
            ),
            dim=0
        ).unsqueeze(0)
    
    def free(self) :
        self._free_board(ctypes.pointer(self))

LP_c_Board = ctypes.POINTER(CBoard)
CBoard._free_board.argtypes = [LP_c_Board]

_next_board = my_library.next_board
_next_board.argtypes = [LP_c_Board, LP_c_int]
_next_board.restype = LP_c_Board

def next_board(cboard: CBoard, directions: list) -> CBoard:
    return _next_board(ctypes.byref(cboard), c_array(directions)).contents