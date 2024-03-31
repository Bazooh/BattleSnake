from utils import *
from MCTS.Game import *

NODE_MAIN_SNAKE = 0
NODE_OTHER_SNAKES = 1
NODE_APPLES = 2

INFINITY = float('inf')

class AlphaBeta:
    def __init__(self, game_state, max_depth: int):
        self.max_depth = max_depth
        self.bounds = np.array([game_state["board"]["width"], game_state["board"]["height"]])
        self.game_info = game_state["game"]
    
    def get_pos(self, pos: typing.Dict) -> np.ndarray:
        return np.array([pos["x"], pos["y"]])
    
    def get_snake_pos(self, snake: typing.Dict) -> list:
        return [self.get_pos(pos) for pos in snake["body"]]
    
    def get_node(self, game_state):
        main_snake = {game_state["you"]["id"]: self.get_snake_pos(game_state["you"])}
        other_snake = {snake["id"]: self.get_snake_pos(snake) for snake in game_state["board"]["snakes"]}
        apples = [self.get_pos(pos) for pos in game_state["board"]["food"]]
        return main_snake, other_snake, apples
    
    def alpha_beta_search(self, game_state) -> str:
        best_val = -INFINITY
        beta = INFINITY

        successors = self.get_successors(self.get_node(game_state), MAIN_PLAYER, as_dict=True)
        best_dir = DEFAULT_MOVE
        for directions, state in successors.items():
            value = self.min_value(state, best_val, beta, 0)
            if value > best_val:
                best_val = value
                best_dir = directions[0]
        
        return best_dir

    def max_value(self, node, alpha: float, beta: float, depth: int) -> float:
        if depth >= self.max_depth:
            return self.get_utility(node, MAIN_PLAYER)

        successors = self.get_successors(node, MAIN_PLAYER)
        if len(successors) <= 0:
            return -INFINITY
        
        value = -INFINITY
        
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta, depth+1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha: float, beta: float, depth: int) -> float:
        if depth >= self.max_depth:
            return self.get_utility(node, MAIN_PLAYER)

        successors: list = self.get_successors(node, OTHER_PLAYER)
        if len(successors) <= 0:
            return INFINITY

        value = INFINITY
        
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta, depth+1))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value
    
    def get_successors(self, node, player: int, as_dict: bool=False) -> typing.Dict | list:
        if player == MAIN_PLAYER:
            states = next_possible_states(node[NODE_MAIN_SNAKE], node[NODE_APPLES], self.bounds)
            for directions, state in states.items():
                if len(state[STATE_SNAKES]) > 0:
                    states[directions] = (state[STATE_SNAKES], node[NODE_OTHER_SNAKES], state[STATE_APPLES])
                else:
                    states.pop(directions)
        else:
            states = next_possible_states(node[NODE_OTHER_SNAKES], node[NODE_APPLES], self.bounds)
            for directions, state in states.items():
                states[directions] = (node[NODE_MAIN_SNAKE], state[STATE_SNAKES], state[STATE_APPLES])
        
        return states if as_dict else states.values()

    def get_utility(self, node, player: int) -> float:
        main_snake = list(node[NODE_MAIN_SNAKE].values())[0]
        other_snakes = node[NODE_OTHER_SNAKES]
        
        return player * (len(main_snake) - sum((len(other_snake) for other_snake in other_snakes.values())))