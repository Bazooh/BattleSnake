# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import sys
from RL.Snake import RLSnake
from UCT.Snake import UCTSnake
from Snake import *
from threading import Thread
from Color import *

def snakes_to_threads(snakes, port):
    threads = []
    for i, snake in enumerate(snakes):
        threads.append(
            Thread(
                target=run_server,
                args=({"info": snake.info, "start": snake.start, "move": snake.move, "end": snake.end}, port + i)
            )
        )
    return threads

def run_threads(threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    port = sys.argv[1] if len(sys.argv) > 1 else 8000
    
    snakes = []
    snakes.append(UCTSnake("UCT/Networks/v_2_1.pt", train=True, should_send_end=True, should_show_tree=False))
    snakes.append(UCTSnake("UCT/Networks/v_2_1.pt", train=False, color=GRAY))
    
    run_threads(snakes_to_threads(snakes, port))

# cd /Users/aymeric/rules && ./battlesnake play -W 11 -H 11 --url http://0.0.0.0:8000 --url http://0.0.0.0:8000  --browser
