from pathlib import Path
import torch
import wandb

from typing import cast
from random import random, randint
from tqdm import tqdm

from Game.cgame import Board, CBoard
from Constants import BATCH_SIZE, DEVICE, GLOBAL_ACTIONS, LR, MAIN_PLAYER, OTHER_PLAYER, UCT_TIME
from Snakes.UCT.Network import Network, SnakeNet, SnakeNetwork
from Snakes.UCT.Train import train, memory, GameMemory
from Snakes.UCT.UCT import UCT


MODEL_PATH = Path("Snakes/UCT/Networks/v_1_0.pt")


def init_game(width: int = 11, height: int = 11, initial_snake_size: int = 3) -> CBoard:
    main_pos = (randint(0, width - 1), randint(0, height - 1))
    while True:
        other_pos = (randint(0, width - 1), randint(0, height - 1))
        if other_pos != main_pos:
            break
    return CBoard.from_board(Board(width, height, [[main_pos] * initial_snake_size, [other_pos] * initial_snake_size]))


def get_random_unoccupied(board: CBoard) -> tuple[int, int]:
    while True:
        pos = (randint(0, cast(int, board.width) - 1), randint(0, cast(int, board.height) - 1))

        if board.snakes_matrix[pos[0]][pos[1]]:
            continue

        return pos


def run_game(model: Network, check_point: Network, spawn_probability: float = 0.15) -> dict[str, float]:
    model.eval()

    board = init_game()

    uct = UCT(board, ["main", "other"], model, max_time=UCT_TIME)
    uct_checkpoint = UCT(board, ["other", "main"], check_point, max_time=UCT_TIME)
    game_memory = GameMemory()

    fps = 0
    snake_size = 0

    turn = 0
    while not board.finished:
        print(board)
        turn += 1

        main_moves = uct.run()
        other_moves = uct_checkpoint.run()

        main_actions_value = uct.root.get_actions_value(MAIN_PLAYER)
        other_actions_value = uct.root.get_actions_value(OTHER_PLAYER)
        game_memory.save_move(board, main_actions_value, other_actions_value, turn)

        fps += (uct.root.size() + uct_checkpoint.root.size()) / UCT_TIME
        n_snakes = int(board.nb_snakes)
        snake_size += sum(len(board.snakes[i].contents) for i in range(n_snakes)) / n_snakes

        new_apples = {get_random_unoccupied(uct.root.board)} if random() < spawn_probability else set()
        
        main_action = GLOBAL_ACTIONS.index(main_moves["main"])
        other_action = GLOBAL_ACTIONS.index(other_moves["other"])
        
        uct.root = uct.root.get_next([main_action, other_action], new_apples)
        uct_checkpoint.root = uct_checkpoint.root.get_next([other_action, main_action], new_apples)

        board = uct.root.board

    assert uct.root.winner is not None

    game_memory.save_into_memory(memory, uct.root.winner, turn)

    return {"turn": turn, "fps": fps / turn, "snake_size": snake_size / turn}


snake_net = SnakeNet().to(DEVICE)
model = SnakeNetwork(snake_net).to(DEVICE)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

model_checkpoint = SnakeNetwork(SnakeNet().to(DEVICE)).to(DEVICE)
model_checkpoint.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(snake_net.parameters(), lr=LR)

wandb.init(project="snake-game", config={"batch_size": BATCH_SIZE, "lr": LR})

for i in tqdm(range(1000)):
    if i % 10 == 0:
        model_checkpoint.load_state_dict(model.state_dict())

    infos = run_game(model, model_checkpoint)

    loss = 0.0
    if len(memory) >= BATCH_SIZE:
        loss = train(snake_net, memory, optimizer)
        torch.save(model.state_dict(), MODEL_PATH)

    wandb.log({"loss": loss} | infos)
