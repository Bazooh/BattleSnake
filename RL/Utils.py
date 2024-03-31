from Constants import *
import random
import torch
import math
from collections import namedtuple, deque
import torch.nn as nn

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def is_colliding_with_snakes(game_state, x, y):
    return {"x": x, "y": y} in game_state["board"]["snakes"]

def is_action_possible(game_state, snakes: set, action) -> bool:
    x, y = game_state["you"]["head"]["x"], game_state["you"]["head"]["y"]
    if ACTIONS[action] == "right":
        x += 1
    elif ACTIONS[action] == "left":
        x -= 1
    elif ACTIONS[action] == "up":
        y += 1
    elif ACTIONS[action] == "down":
        y -= 1
    else:
        print(f"!! WARNING !! the action {action} does not exist")
        return False
    
    return is_in_terrain(x, y) and (x, y) not in snakes

def create_snakes(game_state) -> set:
    snakes = set()
    for snake in game_state["board"]["snakes"]:
        for pos in snake["body"][:-1]:
            snakes.add((pos["x"], pos["y"]))
    return snakes

def possible_actions_mask(game_state) -> torch.Tensor:
    actions = torch.zeros(N_ACTIONS, dtype=torch.float, device=DEVICE)
    snakes = create_snakes(game_state)
    for action in range(N_ACTIONS):
        actions[action] = 1. if is_action_possible(game_state, snakes, action) else 0.
    return actions

def possible_actions(game_state) -> list:
    actions = []
    snakes = create_snakes(game_state)
    for action in range(N_ACTIONS):
        if is_action_possible(game_state, snakes, action):
            actions.append(action)
    return actions

def select_action(game_state, state, policy_net, steps_done=None) -> torch.Tensor:
    sample = random.random()
    
    if steps_done is not None:
        eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-steps_done / EPS_DECAY)
    else:
        eps_threshold = -1
    
    if sample > eps_threshold:
        with torch.no_grad():
            result: torch.Tensor = policy_net(state.unsqueeze(0))[0] * possible_actions_mask(game_state)
            return torch.tensor([result.argmax().item()], dtype=torch.int64, device=DEVICE)
    else:
        actions_possible = possible_actions(game_state)
        if len(actions_possible) <= 0:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = random.choice(actions_possible)
        return torch.tensor([action], dtype=torch.int64, device=DEVICE) 
            

def optimize_model(memory: ReplayMemory, policy_net: torch.nn.Module, target_net: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
    if len(memory) < BATCH_SIZE:
        return 0.
    batch = Transition(*zip(*memory.sample(BATCH_SIZE)))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.stack(tuple(filter(lambda x: x is not None, batch.next_state)))
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def is_in_terrain(x, y) -> bool:
    return 0 <= x < TERRAIN_X_SIZE and 0 <= y < TERRAIN_Y_SIZE

def update_terrain(terrain, i, x, y):
    if is_in_terrain(x, y):
        terrain[i, x, y] = 1.

def you_are_alive(game_state) -> bool:
    your_id = game_state["you"]["id"]
    return your_id in map(lambda snake: snake["id"], game_state["board"]["snakes"])

def game_state_to_tensor(game_state):
    terrain = torch.zeros((5, TERRAIN_X_SIZE, TERRAIN_Y_SIZE), dtype=torch.float, device=DEVICE)
    length = []
    health = []
    
    you = game_state["you"]
    
    length.append(you["length"])
    health.append(you["health"])
    for pos in you["body"][1:]:
        update_terrain(terrain, 0, pos["x"], pos["y"])
    update_terrain(terrain, 1, you["head"]["x"], you["head"]["y"])
    
    your_id = you["id"]
    for snake in game_state["board"]["snakes"]:
        if your_id == snake["id"]: continue
        length.append(snake["length"])
        health.append(snake["health"])
        for pos in snake["body"][1:]:
            update_terrain(terrain, 2, pos["x"], pos["y"])
        update_terrain(terrain, 3, snake["head"]["x"], snake["head"]["y"])
    
    for pos in game_state["board"]["food"]:
        update_terrain(terrain, 4, pos["x"], pos["y"])
    
    return torch.cat((
        terrain.flatten(),
        torch.tensor(length, dtype=torch.float, device=DEVICE) / (11*11),
        torch.tensor(health, dtype=torch.float, device=DEVICE) / 100
    ))
    
    