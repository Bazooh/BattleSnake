from __future__ import annotations
from typing import Tuple, Union, List, Dict
import math
import time
from UCT.Network import Network
import torch
from Constants import *
from Game.cgame import CBoard, next_board

EXPLORATION_CONTROL = math.sqrt(1.5)

class PlayerInvalideValue(Exception):
    def __init__(self, *args: object) -> None:
        print(f"integer: {args[0]}, can't be used for a player")
        super().__init__(*args)


class Tree:
    def __init__(self, action: int = -1, parent: Tree | None = None) -> None :
        self.parent: Union[Tree, None] = parent
        self.action: int = action
        self.children: Union[None, List[Tree]] = None
        self.nb_visited: int = 1
        self.current_player: int = OTHER_PLAYER if parent is None else (1 - parent.current_player)
        """Which player just played on self.board"""
        if self.parent is not None :
            self.board: CBoard = self.parent.board if self.current_player == MAIN_PLAYER else next_board(self.parent.board, [self.parent.action, self.action])
            self.winner: Union[float, None] = self.board.winner if self.board.finished else None
        self.value: Union[float, None] = None
        self.policy: Union[torch.Tensor, None] = None
    
    
    @staticmethod
    def from_game_state(game_state: Dict) -> Tree :
        return Tree.from_cboard(CBoard.from_game_state(game_state))
    
    
    @staticmethod
    def from_cboard(cboard: CBoard) -> Tree :
        tree = Tree()
        tree.board = cboard
        tree.winner = tree.board.winner if tree.board.finished else None
        return tree
    
    
    def get_value(self) -> float :
        """How likely main_player is going to win"""
        if self.winner is not None :
            return self.winner
        
        if self.value is None :
            if self.current_player == MAIN_PLAYER :
                # print(self.parent.policy)
                return self.parent.policy[self.action]
            elif self.parent is None or self.parent.parent is None :
                return 0.5
            return self.parent.parent.policy[N_ACTIONS + self.action]
        return self.value
    
    
    def get_current_player_value(self) -> float :
        """How likely current_player is going to win"""
        if self.current_player == MAIN_PLAYER :
            return self.get_value()
        return 1. - self.get_value()
    
    
    def size(self) -> int :
        if self.winner is not None :
            return 0
        _size = 1
        if self.children is not None :
            _size += sum(child.size() for child in self.children)
        return _size
    
    
    def free(self) :
        if self.children is not None :
            for child in self.children :
                child.free()
        
        if self.parent is None :
            return
        
        if self.current_player != MAIN_PLAYER :
            self.board.free()


def UCB(value: float, step: int, nb_visited: int) -> float :
    """Return the UCB value"""
    if nb_visited == 0 :
        return math.inf
    if step < 2 :
        return 1
    return value + EXPLORATION_CONTROL*math.sqrt( math.log(step) / (nb_visited*(step-1)) )


def action_value(tree: Tree, step: int, action: int) -> float :
    child: Tree = tree.children[action]
    
    if tree.children[action].winner is not None :
        return -math.inf
    
    return UCB(child.get_current_player_value(), step, child.nb_visited) - tree.get_current_player_value()


def selection(tree: Tree, step: int) -> Tree :
    while tree.current_player == MAIN_PLAYER or tree.value is not None :
        best_action: int = max(range(N_ACTIONS), key=lambda action: action_value(tree, step, action))
        tree = tree.children[best_action]
    
    return tree


def expansion(tree: Tree) :
    if tree.current_player == MAIN_PLAYER :
        raise Exception("IN EXPANSION : The current player is the main player")
    
    tree.children = [Tree(i, tree) for i in range(N_ACTIONS)]
    
    for child in tree.children :
        child.children = [Tree(i, child) for i in range(N_ACTIONS)]
        
        if all(map(lambda child: child.winner is not None, child.children)) :
            best_action: int = max(range(N_ACTIONS), key=lambda action: child.children[action].winner)
            child.winner = child.children[best_action].winner


def simulation(tree: Tree, model: Network) :
    if tree.winner is not None :
        return
    
    with torch.no_grad() :
        input_tensor: torch.Tensor = tree.board.to_tensor().to(DEVICE)
        result: torch.Tensor =  model(input_tensor).squeeze(0)
        tree.value = result[0].item()
        tree.policy = result[1:]


def backpropagation(tree: Tree) :
    if tree is None :
        return
    
    tree.nb_visited += 1
    
    f = max if tree.current_player == MAIN_PLAYER else min
    best_action: int = f(range(N_ACTIONS), key=lambda action: tree.children[action].get_value())
    
    if tree.children[best_action].winner is None :
        tree.value = tree.children[best_action].get_value()
    else :
        tree.winner = tree.children[best_action].winner

    backpropagation(tree.parent)


def calc_time(times, idx, func, *args, **kwargs) -> tuple[float, object] :
    start_time = time.time()
    result = func(*args, **kwargs)
    times[idx] += time.time() - start_time
    return result


def UCT(root: Tree, max_time: float, model: Network) -> Dict[int, float] :
    """
    apply the UCT search for the corresponding tree
    """
    step: int = 0
    starting_time = time.time()
    times = [0., 0., 0., 0.]
    
    while time.time() < starting_time + max_time and root.winner is None:
        step += 1
        
        tree = calc_time(times, 0, selection, root, step)
        calc_time(times, 1, expansion, tree)
        calc_time(times, 2, simulation, tree, model)
        calc_time(times, 3, backpropagation, tree.parent)
    
    return {"selection": times[0], "expansion": times[1], "simulation": times[2], "backpropagation": times[3], "total": time.time() - starting_time}
    