from typing import Dict, Tuple, List
from Snake import Snake
from UCT.UCT import Tree, UCT
from UCT.Network import Network
from Constants import *
from UCT.Train import train, memory, GameMemory
import random
import socket
from UCT.Show import show_tree
import os


def get_winner(game_state: Dict) -> float:
    if len(game_state["board"]["snakes"]) >= N_SNAKES or len(game_state["board"]["snakes"]) <= 0 :
        return NO_WINNER
    
    if game_state["board"]["snakes"][0]["id"] == game_state["you"]["id"] :
        return MAIN_PLAYER
    
    return OTHER_PLAYER

class UCTSnake(Snake):
    def __init__(self, path_model, train=False, should_send_end=False, should_show_tree=False, **kwargs):
        super().__init__(**kwargs)
        
        self.path_model = path_model
        self.train = train
        self.should_send_end = should_send_end
        self.should_show_tree = should_show_tree
        self.times = {}
        
        self.model = Network().to(DEVICE)
        if os.path.isfile(path_model):
            self.model.load_state_dict(torch.load(path_model))
        
        if self.train:
            self.steps_done = 0

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)


    def start(self, game_state: Dict):
        self.moves: GameMemory = GameMemory()
        self.tree: Tree = Tree.from_game_state(game_state)
        self.trees: List[Tree] = [self.tree]
        self.end_turn = None
        if os.path.isfile(self.path_model):
            self.model.load_state_dict(torch.load(self.path_model))
        self.model.eval()
        if self.should_send_end:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(('localhost', 8888))

    
    def move(self, game_state: Dict):
        if len(game_state["board"]["snakes"]) < N_SNAKES :
            self.send(game_state, score=0)
            self.end_turn = game_state["turn"]
            return {"move": random.choice(ACTIONS)}
        
        self.tree = Tree.from_game_state(game_state)
        self.trees.append(self.tree)
        
        computing_times = UCT(self.tree, max_time=UCT_TIME, model=self.model)
        for category in computing_times.keys():
            self.times[category] = self.times.get(category, 0) + computing_times[category]
        
        if self.tree.policy is None :
            return {"move": random.choice(ACTIONS)}
        
        actions_value: List[float] = [child.get_value() for child in self.tree.children]
        other_actions_value: List[float] = [
            min(
                child.children[action].get_value()
                for child in self.tree.children
                if child.children is not None
            )
            for action in range(N_ACTIONS)
        ]
        self.moves.save_move(self.tree, actions_value, other_actions_value)
        
        best_action: int = min(range(N_ACTIONS), key=lambda action: actions_value[action])
        
        self.send(game_state, score=actions_value[best_action])
        return {"move": ACTIONS[best_action]}


    def send(self, game_state, score, is_finished=False, loss=None):
        if not self.should_send_end:
            return
        
        message = f"{self.end_turn if is_finished and self.end_turn is not None else game_state['turn']}|{is_finished}|{score}"
        if is_finished:
            message += f"|{loss}"
        message += "\n"
        
        self.sock.sendall(message.encode())
        
        if is_finished:
            self.sock.close()


    def end(self, game_state: Dict):
        winner: float = get_winner(game_state)
        
        self.moves.save_into_memory(memory, winner)
        
        loss = 0.
        if self.train and len(memory) >= BATCH_SIZE:
            self.model.train()
            loss = train(self.model, memory, self.optimizer)
            torch.save(self.model.state_dict(), self.path_model)
        
        if self.should_show_tree:
            show_tree(self.trees[-3], max_depth=None)
        
        for tree in self.trees:
            tree.free()
        
        self.send(game_state, winner, is_finished=True, loss=loss)
        
        print(self.times)