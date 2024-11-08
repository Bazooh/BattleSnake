from Snakes.Snake import Snake
from Snakes.UCT.UCT import UCT
from Snakes.UCT.Network import Network, SnakeNet
from Constants import *
from Snakes.UCT.Train import train, memory, GameMemory
import random
import socket
from Snakes.UCT.Show import show_tree
import os
import cProfile
import threading


def get_winner(game_state: GameState) -> Player | Literal[0]:
    if len(game_state["board"]["snakes"]) >= N_SNAKES or len(game_state["board"]["snakes"]) <= 0:
        return NO_WINNER
    
    if game_state["board"]["snakes"][0]["id"] == game_state["you"]["id"]:
        return MAIN_PLAYER
    
    return OTHER_PLAYER


class UCTSnake(Snake):
    def __init__(self, path_model: str, train: bool = False, profile: bool = False, should_send_end: bool = False, human_policy: bool = False, moves: dict[SnakeId, GlobalAction] = {}, events: list[threading.Event] = [], **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.path_model = path_model
        self.train = train
        self.should_send_end = should_send_end
        self.human_policy = human_policy
        self.moves = moves
        self.events = events
        self.profile = profile
        
        if train:
            self.train_net = SnakeNet().to(DEVICE)
            self.steps_done = 0
            self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=LR)
        
        self.model = Network(self.train_net if self.train else SnakeNet().to(DEVICE)).to(DEVICE)


    def start(self, game_state: GameState) -> None:
        self.memory: GameMemory = GameMemory()
        self.end_turn: int | None = None
        self.moves.clear()
        self.has_shown_tree = False
        
        if os.path.isfile(self.path_model):
            self.model.load_state_dict(torch.load(self.path_model))
        self.model.eval()
        
        if self.should_send_end:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(('localhost', 8888))
        
        self.uct = UCT(game_state, model=self.model, max_time=UCT_TIME, human_policy=self.human_policy)
        
        self.profiler = cProfile.Profile()

    
    def move(self, game_state: GameState) -> Move:
        if self.profile:
            self.profiler.enable()
        
        if len(game_state["board"]["snakes"]) < N_SNAKES:
            self.send(game_state, score=0)
            self.end_turn = game_state["turn"]
            if self.profile:
                self.profiler.disable()
            return {"move": random.choice(GLOBAL_ACTIONS)}
        
        self.uct.update_root(game_state)
        self.uct.run()
        
        main_actions_value = self.uct.root.get_actions_value(MAIN_PLAYER)
        other_actions_value = self.uct.root.get_actions_value(OTHER_PLAYER)
        
        if not self.uct.root.board.finished:
            self.memory.save_move(self.uct.root, main_actions_value, other_actions_value, game_state["turn"])
        
        if not self.has_shown_tree and self.uct.root.has_winner():
            self.has_shown_tree = True
            if self.uct.root.children[0][0] is not None:
                print(self.uct.root.children[0][0].board.finished)
            show_tree(self.uct.root, max_depth=None)
        
        self.send(
            game_state,
            score = self.uct.root.values[0].item() if self.uct.root is not None else 0,
            tree_size = self.uct.root.size()
        )
        
        self.moves.update(self.uct.get_moves())
        for event in self.events:
            event.set()
        
        if self.profile:
            self.profiler.disable()
        
        return {"move": self.moves[game_state["you"]["id"]]}


    def send(self, game_state: GameState, score: float, tree_size: int = 0, is_finished: bool = False, loss: float | None = None) -> None:
        if not self.should_send_end:
            return
        
        message = f"{self.end_turn if is_finished and self.end_turn is not None else game_state['turn']}|{is_finished}|{score}|{tree_size}"
        if is_finished:
            message += f"|{loss}"
        message += "\n"

        self.sock.sendall(message.encode())
        
        if is_finished:
            self.sock.close()


    def end(self, game_state: GameState) -> None:
        if self.profile:
            self.profiler.dump_stats("Snakes/UCT/Profiling/UCT.prof")
        
        winner: float = get_winner(game_state)
        
        if self.train:
            self.memory.save_into_memory(memory, winner, game_state["turn"])
        
        loss = 0.
        if self.train and len(memory) >= BATCH_SIZE:
            self.train_net.train()
            loss = train(self.train_net, memory, self.optimizer)
            print(f"Loss: {loss}")
            torch.save(self.model.state_dict(), self.path_model)
        
        self.send(game_state, winner, is_finished=True, loss=loss)


class UTCSnakeSlave(Snake):
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