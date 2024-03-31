from UCT.Network import Network
from collections import namedtuple, deque
import random
import torch
from Constants import *


def get_symetries(move, value) :
    for inv_player in [True, False] :
        for rotation in [0, 1, 2, 3] :
            for flip in [True, False] :
                board, policy, other_policy = move
                game_board, length, health = board[:-4].unflatten(0, (TERRAIN_DIMS, TERRAIN_X_SIZE, TERRAIN_Y_SIZE)), board[-4:-2], board[-2:]
                
                if inv_player:
                    game_board = torch.index_select(game_board, dim=0, index=torch.tensor([0, 2, 1, 3], dtype=torch.int, device=DEVICE))
                    length = torch.index_select(length, dim=0, index=torch.tensor([1, 0], dtype=torch.int, device=DEVICE))
                    health = torch.index_select(health, dim=0, index=torch.tensor([1, 0], dtype=torch.int, device=DEVICE))
                    value = 1. - value
                
                if flip:
                    game_board = torch.flip(game_board, dims=(1,))
                    policy = torch.index_select(policy, dim=0, index=torch.tensor([0, 3, 2, 1], dtype=torch.int, device=DEVICE))
                    other_policy = torch.index_select(other_policy, dim=0, index=torch.tensor([0, 3, 2, 1], dtype=torch.int, device=DEVICE))
                
                game_board = torch.rot90(game_board, rotation, dims=(1, 2))
                policy = torch.roll(policy, rotation, dims=(0,))
                other_policy = torch.roll(other_policy, rotation, dims=(0,))
                
                yield (
                    torch.cat((game_board.flatten(), length, health)),
                    policy,
                    other_policy,
                    value
                )


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, board, policy, other_policy, value):
        """Save a transition"""
        self.memory.append(Transition(board, policy, other_policy, value))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class GameMemory(object):
    def __init__(self):
        self.moves = []
    
    def save_move(self, tree, actions_value, other_actions_value):
        self.moves.append((
            tree.board.to_tensor().squeeze(0),
            torch.softmax(
                torch.tensor(
                    actions_value,
                    dtype=torch.float,
                    device=DEVICE
                ),
                dim=0
            ),
            torch.softmax(
                torch.tensor(
                    other_actions_value,
                    dtype=torch.float,
                    device=DEVICE
                ),
                dim=0
            )
        ))
    
    def save_into_memory(self, memory: ReplayMemory, winner: float):
        for move in self.moves:
            for symetric_board, symetric_policy, other_symetric_policy, symetric_winner in get_symetries(move, winner):
                memory.push(symetric_board, symetric_policy, other_symetric_policy, torch.tensor(symetric_winner, dtype=torch.float, device=DEVICE))
    
    def __len__(self):
        return len(self.moves)


Transition = namedtuple('Transition', ('board', 'policy', 'other_policy', 'value'))
memory = ReplayMemory(MEMORY_SIZE)


def train(model: Network, memory: ReplayMemory, optimizer):
    cumul_loss = 0.
    
    for epoch in range(EPOCHS):
        batch = Transition(*zip(*memory.sample(BATCH_SIZE)))
        
        batch_board = torch.stack(batch.board)
        batch_policy = torch.stack(batch.policy)
        batch_other_policy = torch.stack(batch.other_policy)
        batch_value = torch.stack(batch.value).unsqueeze(1)
        
        optimizer.zero_grad()
        output = model(batch_board)
        criterion = torch.nn.MSELoss()
        
        loss = criterion(output, torch.cat((batch_value, batch_policy, batch_other_policy), dim=1))
        loss.backward()
        optimizer.step()
        
        cumul_loss += loss.item()
    
    return cumul_loss / EPOCHS