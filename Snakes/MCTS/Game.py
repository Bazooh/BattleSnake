from copy import deepcopy
from Utils.utils import *
import typing
from itertools import product

STATE_SNAKES = 0
STATE_APPLES = 1

NODE_MAIN_SNAKE = 0
NODE_OTHER_SNAKES = 1
NODE_APPLES = 2

class Game():
    def __init__(self, starting_game_state):
        self.starting_game_state = starting_game_state
        self.number_of_snakes = len(starting_game_state["board"]["snakes"])
        self.actions = list(product(DIRECTIONS.values(), repeat=self.number_of_snakes-1))

    def getInitBoard(self):
        def get_pos(pos: typing.Dict) -> np.ndarray:
            return np.array([pos["x"], pos["y"]])
    
        def get_snake_pos(snake: typing.Dict) -> list:
            return [get_pos(pos) for pos in snake["body"]]
    
        main_snake = {self.starting_game_state["you"]["id"]: get_snake_pos(self.starting_game_state["you"])}
        other_snake = {snake["id"]: get_snake_pos(snake) for snake in self.starting_game_state["board"]["snakes"]}
        apples = [get_pos(pos) for pos in self.starting_game_state["board"]["food"]]
        return main_snake, other_snake, apples

    def getBoardSize(self):
        return np.array([self.starting_game_state["board"]["width"], self.starting_game_state["board"]["height"]])

    def getActionSize(self):
        return len(self.actions)

    def getNextState(self, board, player, action):
        grid = get_grid({**board[NODE_MAIN_SNAKE], **board[NODE_OTHER_SNAKES]}, self.getBoardSize())
        
        if player == MAIN_PLAYER:
            new_snake, apples = move_snakes(grid, board[NODE_MAIN_SNAKE], board[NODE_APPLES], [list(DIRECTIONS.values())[action]])    
            return (new_snake, board[NODE_OTHER_SNAKES], apples), -player
        
        else:
            new_snakes, apples = move_snakes(grid, board[NODE_OTHER_SNAKES], board[NODE_APPLES], self.actions[action])    
            return (board[NODE_MAIN_SNAKE], new_snakes, apples), -player

    def getValidMoves(self, board, player):
        grid = get_grid({**board[NODE_MAIN_SNAKE], **board[NODE_OTHER_SNAKES]}, self.getBoardSize())
        valid_moves = []
        
        if player == MAIN_PLAYER:
            main_snake = list(board[NODE_MAIN_SNAKE].values())[0]
            for direction in DIRECTIONS.values():
                valid_moves.append(int(is_move_valid(grid, main_snake[SNAKE_HEAD] + direction)))
            return valid_moves

        else:
            for snake in board[NODE_OTHER_SNAKES].values():
                for direction in DIRECTIONS.values():
                    valid_moves.append(int(is_move_valid(grid, snake[SNAKE_HEAD] + direction)))
            return valid_moves

    def getGameEnded(self, board, player):
        if board[NODE_MAIN_SNAKE] == board[NODE_OTHER_SNAKES] == {}:
            return 1e-4
        
        elif board[NODE_MAIN_SNAKE] == {}:
            return -player
        
        elif board[NODE_OTHER_SNAKES] == {}:
            return player
        
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return str(board)
    
    def eval(self, board, player):
        main_snake = list(board[NODE_MAIN_SNAKE].values())[0]
        other_snakes = board[NODE_OTHER_SNAKES]
        
        return player * (len(main_snake) - sum((len(other_snake) for other_snake in other_snakes.values())))

def next_possible_states(snakes: typing.Dict, apples: list, bounds: np.ndarray) -> typing.Dict:
    states = {}
    
    grid = get_grid(snakes, bounds)
    for directions in product(*(valid_moves(grid, snake[SNAKE_HEAD]) for snake in snakes.values())):
        states[directions] = move_snakes(grid, snakes, apples, directions)
    
    return states


def move_snakes(grid: np.ndarray, snakes: typing.Dict, apples: list, directions: list) -> tuple[typing.Dict, list]:
    next_apples = deepcopy(apples)
    next_snakes = {}

    for snake_id, direction in zip(snakes, directions):
        snake = snakes[snake_id]
        next_head = get_next_head(snake[SNAKE_HEAD], direction)

        if not is_in_grid(grid, next_head):
            continue

        if collides_with_other_snake(grid, next_head):
            continue
        
        next_snake = move_snake(deepcopy(snake), next_head)

        if any(np.array_equal(next_head, apple) for apple in apples):
            next_apples = list(filter(lambda x: not np.array_equal(x, next_head), next_apples))
            extend_snake(next_snake)

        next_snakes[snake_id] = next_snake
    
    remove_colliding_heads(next_snakes)

    return next_snakes, next_apples


def get_next_head(head: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.array([head[X] + direction[X], head[Y] + direction[Y]])


def remove_colliding_heads(snakes: typing.Dict):
    snake_heads = {}
    
    for snake_id, snake in snakes.items():
        val = snake_heads.get((snake[SNAKE_HEAD][X], snake[SNAKE_CASE][Y]), [])
        val.append(snake_id)
        snake_heads[(snake[SNAKE_HEAD][X], snake[SNAKE_CASE][Y])] = val
    
    for list_id in snake_heads.values():
        if len(list_id) > 1:
            lengths = {snake_id: len(snakes[snake_id]) for snake_id in list_id}
            max_value = max(lengths.values())
            args_max = [key for key, value in lengths.items() if value == max_value]
            
            item_to_remove = set(lengths)
            if len(args_max) == 1:
                item_to_remove.remove(args_max[0])
            
            for item in item_to_remove:
                snakes.pop(item)


def extend_snake(snake: list):
    snake.append(snake[-1])


def move_snake(snake: list, next_head: np.ndarray) -> list:
    snake.insert(SNAKE_HEAD, next_head)
    snake.pop()
    return snake
