import random
import socket
import os
import cProfile
import threading
import torch
import torch.nn as nn

from typing import Literal

from Game.cgame import CBoard
from Snakes.Snake import Snake
from Snakes.UCT.UCT import UCT
from Snakes.UCT.Network import Evaluation, HumanEval, SnakeNet, SnakeNetwork
from Snakes.UCT.Train import train, memory, GameMemory

# from Snakes.UCT.Show import show_tree
from Constants import (
    GameState,
    Player,
    SnakeId,
    GlobalAction,
    Move,
    N_SNAKES,
    DEVICE,
    LR,
    UCT_TIME,
    NO_WINNER,
    MAIN_PLAYER,
    OTHER_PLAYER,
    GLOBAL_ACTIONS,
    BATCH_SIZE,
)


def get_winner(game_state: GameState) -> Player | Literal[0]:
    if len(game_state["board"]["snakes"]) >= N_SNAKES or len(game_state["board"]["snakes"]) <= 0:
        return NO_WINNER

    if game_state["board"]["snakes"][0]["id"] == game_state["you"]["id"]:
        return MAIN_PLAYER

    return OTHER_PLAYER


class UCTSnake(Snake):
    def __init__(
        self,
        path_model: str | None = None,
        train: bool = False,
        profile: bool = False,
        should_send_end: bool = False,
        human_policy: bool = False,
        moves: dict[SnakeId, GlobalAction] = {},
        events: list[threading.Event] = [],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert not (train and human_policy), "Cannot train the human policy"
        assert human_policy or path_model is not None, "Need a model to load"

        self.path_model = path_model
        self.train = train
        self.should_send_end = should_send_end
        self.moves = moves
        self.events = events
        self.profile = profile
        self.human_policy = human_policy
        self.model: Evaluation

        if self.should_send_end:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(("localhost", 8888))

        if train:
            self.train_net = SnakeNet().to(DEVICE)
            self.steps_done = 0
            self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=LR)

        if human_policy:
            self.model = HumanEval()
        else:
            self.model = SnakeNetwork(self.train_net if self.train else SnakeNet().to(DEVICE)).to(DEVICE)

    def start_from_board(self, board: CBoard, ids: list[SnakeId]) -> None:
        self.memory = GameMemory()
        self.end_turn: int | None = None
        self.moves.clear()
        self.has_shown_tree = False

        if isinstance(self.model, nn.Module):
            assert self.path_model is not None, "Need a path to load the model"

            self.model.eval()
            if os.path.isfile(self.path_model):
                self.model.load_state_dict(torch.load(self.path_model, weights_only=True))

        self.uct = UCT(board, ids, model=self.model, max_time=UCT_TIME)

        self.profiler = cProfile.Profile()

    def start(self, game_state: GameState) -> None:
        self.start_from_board(
            CBoard.from_game_state(game_state),
            [game_state["you"]["id"]]
            + [snake["id"] for snake in game_state["board"]["snakes"] if snake["id"] != game_state["you"]["id"]],
        )

    def run_uct(self, turn: int) -> dict[SnakeId, GlobalAction]:
        moves = self.uct.run()

        main_actions_value = self.uct.root.get_actions_value(MAIN_PLAYER)
        other_actions_value = self.uct.root.get_actions_value(OTHER_PLAYER)

        if not self.uct.root.board.finished:
            self.memory.save_move(self.uct.root.board, main_actions_value, other_actions_value, turn)

        return moves

    def move(self, game_state: GameState) -> Move:
        if self.profile:
            self.profiler.enable()

        if len(game_state["board"]["snakes"]) < N_SNAKES:
            self.__send(game_state["turn"], score=0)
            self.end_turn = game_state["turn"]
            if self.profile:
                self.profiler.disable()
            return {"move": random.choice(GLOBAL_ACTIONS)}

        self.uct.update_root(
            CBoard.from_game_state(game_state), set((apple["x"], apple["y"]) for apple in game_state["board"]["food"])
        )
        moves = self.run_uct(game_state["turn"])

        print(self.uct.root.board)

        # if not self.has_shown_tree and self.uct.root.has_winner():
        #     self.has_shown_tree = True
        #     print("finished")
        #     show_tree(self.uct.root, max_depth=None)

        self.__send(
            game_state["turn"],
            score=self.uct.root.values[0].item() if self.uct.root is not None else 0,
            tree_size=self.uct.root.size(),
        )

        self.moves.update(moves)
        for event in self.events:
            event.set()

        if self.profile:
            self.profiler.disable()

        print(game_state["you"]["id"], self.moves)

        return {"move": self.moves[game_state["you"]["id"]]}

    def __send(
        self, turn: int, score: float, tree_size: int = 0, is_finished: bool = False, loss: float | None = None
    ) -> None:
        if not self.should_send_end:
            return

        message = f"{self.end_turn if is_finished and self.end_turn is not None else turn}|{is_finished}|{score}|{tree_size}"
        if is_finished:
            message += f"|{loss}"
        message += "\n"

        self.sock.sendall(message.encode())

        if is_finished:
            self.sock.close()

    def end_from_winner(self, winner: float, turn: int) -> None:
        if self.profile:
            self.profiler.dump_stats("Snakes/UCT/Profiling/UCT.prof")

        if self.train:
            self.memory.save_into_memory(memory, winner, turn)

        loss = 0.0
        if self.train and len(memory) >= BATCH_SIZE:
            assert self.path_model is not None, "Need a path to save the model"
            assert isinstance(self.model, nn.Module), "Need a trainable model"

            self.train_net.train()
            loss = train(self.train_net, memory, self.optimizer)
            print(f"Loss: {loss}")
            torch.save(self.model.state_dict(), self.path_model)

        self.__send(turn, winner, is_finished=True, loss=loss)

    def end(self, game_state: GameState) -> None:
        self.end_from_winner(get_winner(game_state), game_state["turn"])


class UCTSnakeSlave(Snake):
    def __init__(self, moves: dict[SnakeId, GlobalAction], event: threading.Event, **kwargs):
        super().__init__(**kwargs)
        self.moves = moves
        self.event = event

    def start(self, game_state: GameState) -> None:
        self.event.clear()

    def move(self, game_state: GameState) -> Move:
        if len(game_state["board"]["snakes"]) < N_SNAKES:
            return {"move": random.choice(GLOBAL_ACTIONS)}

        self.event.wait()
        self.event.clear()

        return {"move": self.moves[game_state["you"]["id"]]}
