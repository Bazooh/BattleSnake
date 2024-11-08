from Snakes.Snake import Snake
from Snakes.RL.Utils import *
from Snakes.RL.Network import Network
import torch
import torch.optim as optim
from Snakes.RL.Reward import get_state_reward
import os
import socket
from Constants import *

memory = ReplayMemory(MEMORY_SIZE)
torch.autograd.set_detect_anomaly(True)

class RLSnake(Snake):
    def __init__(self, path_policy, train=False, should_send_end=False, color=None):
        super().__init__()
        
        self.path_policy = path_policy
        self.train = train
        self.should_send_end = should_send_end
        if color is not None:
            self.color = color
        
        self.policy_net = Network().to(DEVICE)
        if os.path.isfile(path_policy):
            self.policy_net.load_state_dict(torch.load(path_policy))
        
        if self.train:
            self.steps_done = 0
            
            self.target_net = Network().to(DEVICE)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
    
    def start(self, game_state):
        if not self.train:
            if os.path.isfile(self.path_policy):
                self.policy_net.load_state_dict(torch.load(self.path_policy))
        
        self.last_game_state = None
        self.last_action = None
    
    def update_net(self):
        loss = optimize_model(memory, self.policy_net, self.target_net, self.optimizer)

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss
    
    def update_memory(self, game_state, state):
        if self.last_game_state is None:
            return

        reward = torch.tensor([get_state_reward(game_state)], dtype=torch.float, device=DEVICE)
        memory.push(self.last_game_state, self.last_action, state, reward)
    
    def send_end(self, game_state, loss):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8888))
        sock.sendall(f"{game_state['turn']}|{loss}".encode())
        sock.close()
    
    def end(self, game_state):
        if self.train:
            self.update_memory(game_state, state=None)
            loss = 0.
            for _ in range(EPOCHS):
                loss += self.update_net()
            
            torch.save(self.policy_net.state_dict(), self.path_policy)
        
        if self.should_send_end:
            self.send_end(game_state, (loss / EPOCHS) if self.train else None)
    
    def move(self, game_state):
        if len(game_state["board"]["snakes"]) <= 1:
            # WTF ????
            return {"move": "up"}
        
        state = game_state_to_tensor(game_state)
        
        if self.train:
            self.update_memory(game_state, state)
        
        your_head = game_state["you"]["head"]["x"], game_state["you"]["head"]["y"]
        action = select_action(game_state, state, self.policy_net, self.steps_done if self.train else None)
        
        if self.train:
            self.steps_done += 1
            self.last_game_state = state
            self.last_action = action
        
        return {"move": GLOBAL_ACTIONS[action.item()]}