import typing

from abc import ABC, abstractmethod

from Constants import GameState, Move, GlobalAction


class Snake(ABC):
    color = "#f57f18"
    head = "scarf"
    tail = "rbc-necktie"

    def __init__(self, **kwargs):
        self.color = kwargs.get("color", self.color)

    def info(self) -> typing.Dict:
        """See https://docs.battlesnake.com/api/requests/info for more info"""
        return {
            "apiversion": "1",
            "author": "Bazooh",
            "color": self.color,
            "head": self.head,
            "tail": self.tail,
        }

    def start(self, game_state: GameState):
        """Start is called when your Battlesnake begins a game"""
        ...

    def end(self, game_state: GameState):
        """End is called when your Battlesnake finishes a game"""
        ...

    @abstractmethod
    def move(self, game_state: GameState) -> Move:
        """Called each turn
        Return a dict of type {"move": movement: str}
            where movement is one of ["right", "left", "top", "down"]
        """
        ...


class StupidSnake(Snake):
    directions: list[GlobalAction] = ["right", "up", "left", "down"]

    def __init__(self, start_idx: int = -1) -> None:
        """Makes a circle with the directions

        Args:
            start_idx (int, optional): the direction the snakes starts with. Defaults to -1 (probably right)
        """
        super().__init__()
        self.current_idx = start_idx

    def move(self, game_state: GameState) -> Move:
        self.current_idx += 1
        self.current_idx %= 4
        return {"move": self.directions[self.current_idx]}

    def end(self, game_state: GameState) -> None:
        print(game_state)
