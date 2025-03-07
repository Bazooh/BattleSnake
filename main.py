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

from threading import Thread, Event
from Server import run_server

from Utils.Color import GRAY

from Snakes.UCT.Snake import UCTSnake, UCTSnakeSlave
from Snakes.AlphaBeta.alpha_beta import AlphaBetaSnake


def snakes_to_threads(snakes, port):
    threads = []
    for i, snake in enumerate(snakes):
        threads.append(
            Thread(
                target=run_server,
                args=({"info": snake.info, "start": snake.start, "move": snake.move, "end": snake.end}, port + i),
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
    port = sys.argv[1] if len(sys.argv) > 1 else 8000

    snakes = []
    moves = {}
    events = [Event()]
    snakes.append(
        UCTSnake(
            "Snakes/UCT/Networks/v_1_0.pt",
            train=False,
            profile=False,
            human_policy=False,
            should_send_end=False,
            moves=moves,
            events=events,
        )
    )
    # snakes.append(UCTSnakeSlave(moves=moves, event=events[0], color=GRAY))

    # snakes.append(AlphaBetaSnake(color=GRAY))
    # snakes.append(AlphaBetaSnake())

    run_threads(snakes_to_threads(snakes, port))

# /Users/aymeric/rules/battlesnake play -W 11 -H 11 --url http://0.0.0.0:8000 --url http://0.0.0.0:8001  --browser
