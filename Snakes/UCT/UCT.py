from __future__ import annotations
from typing import Generator, cast
import math
import time
from Snakes.UCT.Network import Network
import torch
from Constants import *
from Game.cgame import CBoard, next_board
import random
from Utils.utils import get_global_direction_idx
from scipy.optimize import linprog
import numpy as np


EXPLORATION_CONTROL = 0.5
RANDOM_EPSILON = 1e-8


class PlayerInvalideValue(Exception):
    def __init__(self, *args: tuple) -> None:
        print(f"integer: {args[0]}, can't be used for a player")
        super().__init__(*args)


class Root:
    def __init__(self, board: CBoard, values: torch.Tensor, policies: torch.Tensor) -> None:
        self.board = board
        self.values = values
        self.policies = policies
        self.children: list[list[Tree | None]] = [[None, None, None], [None, None, None], [None, None, None]]
        """children[main_action][other_action]"""
        
        self.winner: Player | None = cast(Player, self.board.winner) if self.board.finished else None

        self.playable_actions: list[list[LocalAction]] = [
            [cast(LocalAction, action) for action, playable in enumerate(self.board.snakes[0].contents.playable_actions) if playable],
            [cast(LocalAction, action) for action, playable in enumerate(self.board.snakes[1].contents.playable_actions) if playable],
        ]
        """0 is the main player, 1 is the other player"""
        
        self.nb_visited: int = 0
        self.actions_nb_visited: list[list[int]] = [[0, 0, 0], [0, 0, 0]]
    
    
    @classmethod
    def from_game_state(cls, game_state: GameState, policies: torch.Tensor | None = None) -> Root:
        return cls.from_cboard(CBoard.from_game_state(game_state), policies)
    
    
    @classmethod
    def from_cboard(cls, board: CBoard, values: torch.Tensor | None = None, policies: torch.Tensor | None = None) -> Root:
        if values is None:
            values = torch.tensor([NO_WINNER_PENALTY, NO_WINNER_PENALTY], dtype=torch.float, device=DEVICE)
        if policies is None:
            policies = torch.ones((2, N_LOCAL_ACTIONS), dtype=torch.float, device=DEVICE) / N_LOCAL_ACTIONS
        return cls(board, values, policies)
    
    
    @classmethod
    def from_tree(cls, tree: Tree) -> Root:
        root = cls(tree.board, tree.values, tree.policies)
        root.children = tree.children
        return root
    
    
    def create_child(self, actions: tuple[LocalAction, LocalAction]) -> Tree:
        assert self.children[actions[0]][actions[1]] is None
        
        tree = Tree(
            actions = actions,
            parent = self,
            board = next_board(self.board, actions),
            values = torch.tensor([NO_WINNER_PENALTY, NO_WINNER_PENALTY], dtype=torch.float, device=DEVICE),
            policies = torch.ones((2, N_LOCAL_ACTIONS), dtype=torch.float, device=DEVICE) / N_LOCAL_ACTIONS
        )
        self.children[actions[0]][actions[1]] = tree
        
        return tree
    
    
    def size(self) -> int:
        size = 1
        for row in self.children:
            for child in row:
                if child is not None:
                    size += child.size()
        
        return size
    
    
    def depth(self) -> int:
        depth = 1
        for row in self.children:
            for child in row:
                if child is not None:
                    depth = max(depth, 1 + child.depth())
        
        return depth
    
    
    def get_child(self, main_action: LocalAction, other_action: LocalAction, pov: Player = MAIN_PLAYER) -> Tree | None:
        return self.children[main_action][other_action] if pov == MAIN_PLAYER else self.children[other_action][main_action]
    
    
    def get_children(self) -> Generator[Tree, None, None]:
        for row in self.children:
            for child in row:
                if child is not None:
                    yield child
    
    
    def mean_actions_value(self, player: Player, other_action: LocalAction, default: float = 0, pov: Player = MAIN_PLAYER) -> float:
        """The mean value of the actions for the player, given the other player's action.\n
        View by pov: 1 is in favor of pov and -1 in its defavor\n
        If the action is not visited, return the default value."""
        actions = [
            child.values[0 if pov == MAIN_PLAYER else 1]
                for action in self.playable_actions[0 if player == MAIN_PLAYER else 1]
                if (child := self.children[action if player == MAIN_PLAYER else other_action][other_action if player == MAIN_PLAYER else action]) is not None
        ]
        
        if len(actions) == 0:
            return default

        return float(sum(actions) / len(actions))
    
    
    def get_actions_value(self, player: Player) -> tuple[float, float, float]:
        """1 is an action that leads to a win, -1 to a loss, 0 to a draw, and -1 to an unvisited action"""
        return cast(tuple[float, float, float], tuple(
            self.mean_actions_value(-player, action, default = -1, pov = player) if action in self.playable_actions[0 if player == MAIN_PLAYER else 1] else -1 
            for action in range(N_LOCAL_ACTIONS)
        ))
    
    
    def best_actions(self) -> tuple[LocalAction, LocalAction]:
        """Compute the best actions for the main and the other player"""
        rho = EXPLORATION_CONTROL * math.sqrt(self.nb_visited)
        
        main_best_action = max(
            self.playable_actions[0],
            key = lambda main_action: self.mean_actions_value(OTHER_PLAYER, main_action, pov=MAIN_PLAYER) \
                + rho*self.policies[0][main_action].item() / (1 + self.actions_nb_visited[0][main_action]) \
                + RANDOM_EPSILON*random.random()
        )
        other_best_action = max(
            self.playable_actions[1],
            key = lambda other_action: self.mean_actions_value(MAIN_PLAYER, other_action, pov=OTHER_PLAYER) \
                + rho*self.policies[1][other_action].item() / (1 + self.actions_nb_visited[1][other_action]) \
                + RANDOM_EPSILON*random.random()
        )
        
        return main_best_action, other_best_action
    
    
    def all_actions_have_winner(self) -> bool:
        return all(
            (child := self.children[main_action][other_action]) is not None and child.has_winner()
                for main_action in self.playable_actions[0] for other_action in self.playable_actions[1]
        )
    
    
    def has_winner(self) -> bool:
        return self.winner is not None
    
    
    def compute_winner(self) -> bool:
        """Compute the winner if there is one, and return True if there is a winner"""
        if self.has_winner():
            return True
        
        if any(
            all(
                (child := self.children[main_action][other_action]) is not None and child.winner == MAIN_PLAYER
                    for other_action in self.playable_actions[1]
            ) for main_action in self.playable_actions[0]
        ):
            self.winner = MAIN_PLAYER
            return True
        
        if any(
            all(
                (child := self.children[main_action][other_action]) is not None and child.winner == OTHER_PLAYER
                    for main_action in self.playable_actions[0]
            ) for other_action in self.playable_actions[1]
        ):
            self.winner = OTHER_PLAYER
            return True

        if self.all_actions_have_winner():
            self.winner = -linprog(
                c = [0 for _ in self.playable_actions[0]] + [-1], # * -1 or 1 ?
                A_ub = [
                    [-self.children[main_action][other_action].winner for main_action in self.playable_actions[0]] + [1] # type: ignore
                        for other_action in self.playable_actions[1]
                ],
                b_ub = [0 for _ in self.playable_actions[1]],
                A_eq = [[1 for _ in self.playable_actions[0]] + [0]],
                b_eq = [1],
                bounds = [(0, 1) for _ in self.playable_actions[0]] + [(None, None)],
            ).fun
            return True
        
        return False


class Tree(Root):
    def __init__(
        self,
        board: CBoard,
        actions: tuple[LocalAction, LocalAction],
        parent: Tree | Root,
        values: torch.Tensor,
        policies: torch.Tensor
    ) -> None:
        super().__init__(board, values, policies)
        
        self.parent = parent
        self.actions = actions
        self.values = values
    
    
    def update_apples(self, apples: set[Pos]) -> None:
        for apple in apples:
            self.board.apples_matrix[apple[0]][apple[1]] = 1
            if self.board.snakes_matrix[apple[0]][apple[1]] != 0:
                self.reset()
                return
        
        for child in self.get_children():
            child.update_apples(apples)
    
    
    def reset(self) -> None:
        self.parent.children[self.actions[0]][self.actions[1]] = None


class UCT:
    def __init__(self, game_state: GameState, model: Network, max_time: float, human_policy: bool = False) -> None:
        self.model = model
        self.max_time = max_time
        self.human_policy = human_policy
        self.init_root(game_state)
    
    
    def selection(self, root: Root) -> tuple[Tree | Root, LocalAction, LocalAction]:
        main_best_action, other_best_action = root.best_actions()
        tree = next_tree = root.children[main_best_action][other_best_action]
        
        if tree is None:
            return root, main_best_action, other_best_action
        
        while next_tree is not None and not next_tree.has_winner():
            tree = next_tree
            
            main_best_action, other_best_action = tree.best_actions()
            next_tree = tree.children[main_best_action][other_best_action]
        
        return tree, main_best_action, other_best_action
            


    def expansion(self, tree: Tree | Root, main_best_action: LocalAction, other_best_action: LocalAction) -> Tree:
        # if the child already exists (it should be a leaf node), return it
        # assert tree.children[main_best_action][other_best_action] is not None
        
        if (child := tree.children[main_best_action][other_best_action]) is not None:
            return child
        
        return tree.create_child((main_best_action, other_best_action))


    def simulate(self, board: CBoard) -> tuple[torch.Tensor, torch.Tensor]:
        if board.finished:
            policies = torch.zeros((2, N_LOCAL_ACTIONS), dtype=torch.float, device=DEVICE)
            
            if (winner := cast(Player, board.winner)) == NO_WINNER:
                return torch.tensor([NO_WINNER_PENALTY, NO_WINNER_PENALTY], dtype=torch.float, device=DEVICE), policies
            return torch.tensor([winner, -winner], dtype=torch.float, device=DEVICE), policies
        
        tensor = board.to_tensors(DEVICE)
        
        if self.human_policy:
            len_0 = tensor[1][0][0].item()
            len_1 = tensor[1][1][0].item()
            
            value = (len_0 - len_1) / (cast(int, board.width) * cast(int, board.height))
            value = torch.tensor([value, -value], dtype=torch.float, device=DEVICE)
            
            policies = tensor[1][:, 1:]
            
            policies[0, :] = policies[0, :].max() - policies[0, :]
            policies[1, :] = policies[1, :].max() - policies[1, :]
            
            policies[0, :] /= policies[0, :].sum()
            policies[1, :] /= policies[1, :].sum()
            
            return value, policies
        
        with torch.no_grad():
            return self.model(*tensor)


    def simulation(self, tree: Tree) -> None:
        tree.nb_visited += 1
        tree.values, tree.policies = self.simulate(tree.board)
    

    def backpropagation(self, tree: Tree | Root, actions: tuple[LocalAction, LocalAction]) -> None:
        assert (child := tree.children[actions[0]][actions[1]]) is not None
        
        tree.nb_visited += 1
        for i, action in enumerate(actions):
            tree.actions_nb_visited[i][action] += 1
        
        if child.has_winner() and tree.compute_winner():
            assert tree.winner is not None
            tree.values = torch.tensor([tree.winner, -tree.winner], dtype=torch.float, device=DEVICE)
        else:
            tree.values = (tree.values * (tree.nb_visited - 1) + child.values) / tree.nb_visited
        
        if isinstance(tree, Tree):
            self.backpropagation(tree.parent, tree.actions)


    def run(self) -> None:
        """
        apply the UCT search from self.root
        """
        step: int = 0
        starting_time = time.time()
        
        self.root.values, self.root.policies = self.simulate(self.root.board)
        
        while time.time() < starting_time + self.max_time and self.root.winner is None:
            step += 1
            
            tree, main_best_action, other_best_action = self.selection(self.root)
            tree = self.expansion(tree, main_best_action, other_best_action)
            self.simulation(tree)
            self.backpropagation(tree.parent, (main_best_action, other_best_action))
    
    
    def init_root(self, game_state: GameState) -> None:
        self.root = Root.from_game_state(game_state)
        self.ids: list[str] = [game_state["you"]["id"]] + [snake["id"] for snake in game_state["board"]["snakes"] if snake["id"] != game_state["you"]["id"]]
    
    
    def update_root(self, game_state: GameState) -> None:
        if self.root.board.turn == 0:
            self.init_root(game_state)
            return
        
        if game_state["turn"] == self.root.board.turn:
            raise Exception(f"Trying to update the game, but it has already been updated to turn {game_state['turn']}")
        
        main_snake = game_state["you"]
        other_snake = [snake for snake in game_state["board"]["snakes"] if snake["id"] != main_snake["id"]][0]
        
        main_last_global_action = self.root.board.snakes[0].contents.global_direction
        other_last_global_action = self.root.board.snakes[1].contents.global_direction
        main_global_action = get_global_direction_idx(main_snake)
        other_global_action = get_global_direction_idx(other_snake)
        
        main_action = (main_global_action - main_last_global_action + 1) & 0b11
        other_action = (other_global_action - other_last_global_action + 1) & 0b11
        
        if self.root.children[main_action][other_action] is None:
            self.root.create_child((main_action, other_action))
        
        self.root = Root.from_tree(self.root.children[main_action][other_action])
        apples = set((apple["x"], apple["y"]) for apple in game_state["board"]["food"])
        for child in self.root.get_children():
            child.update_apples(apples)
    
    
    def get_moves(self) -> dict[SnakeId, GlobalAction]:
        moves = {}
        
        for player, id in enumerate(self.ids):
            actions_value = self.root.get_actions_value(MAIN_PLAYER if player == 0 else OTHER_PLAYER)
            
            arg_max = cast(LocalAction, np.argmax(actions_value))
            max_value = actions_value[arg_max]
            
            best_actions = cast(list[LocalAction], [i for i, value in enumerate(actions_value) if value == max_value])
            if len(best_actions) > 1 and len(self.root.playable_actions[1 - player]) >= 1:
                if max_value == -1:
                    # If this is a loss anyway, try to hold on as long as possible with what was computed
                    # TODO : this is not optimal, we should continue the search even if the game is sure to end
                    arg_max = max(
                        best_actions,
                        key = lambda main_action: min(
                            child.depth() if (child := self.root.get_child(main_action, other_action, MAIN_PLAYER if player == 0 else OTHER_PLAYER)) is not None else 0
                                for other_action in self.root.playable_actions[1 - player]
                        )
                    )
                else:
                    arg_max = random.choice(best_actions)
            
            global_action = (cast(int, self.root.board.snakes[player].contents.global_direction) + arg_max - 1) & 0b11 # % 4
            moves[id] = GLOBAL_ACTIONS[global_action]
        
        return moves