from RL.Utils import *

def get_state_reward(game_state):
    if you_are_alive(game_state):
        if len(game_state["board"]["snakes"]) == 1:
            return 1.
        return 0.
    return 0.