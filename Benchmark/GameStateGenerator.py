import random


def GetFreePosition(snake_matrix: list[list[bool]], board_height: int, board_width: int) -> dict[str, int]:
    while True:
        x = random.randint(0, board_height - 1)
        y = random.randint(0, board_width - 1)
        if not snake_matrix[x][y]:
            snake_matrix[x][y] = True
            return {"x": x, "y": y}


def GetFreePositionNextTo(snake_matrix: list[list[bool]], board_height: int, board_width: int, x: int, y: int) -> dict[str, int] | None:
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random.shuffle(directions)
    
    for dx, dy in directions:
        new_x = x + dx
        new_y = y + dy
        if 0 <= new_x < board_height and 0 <= new_y < board_width and not snake_matrix[new_x][new_y]:
            snake_matrix[new_x][new_y] = True
            return {"x": new_x, "y": new_y}
    
    return None


def GenerateRandomSnake(snake_matrix: list[list[bool]], id: str, board_height: int = 11, board_width: int = 11, length: int = 3) -> dict[str]:
    head: dict[str, int] = GetFreePosition(snake_matrix, board_height, board_width)
    body = [head]
    for _ in range(length - 1):
        new_position = GetFreePositionNextTo(snake_matrix, board_height, board_width, body[-1]["x"], body[-1]["y"])
        if new_position is None:
            break
        body.append(new_position)
    
    return {
        "id": id,
        "health": random.randint(1, 100),
        "body": body,
        "head": head,
        "length": length,
    }


def GenerateRandomGameState(
    *,
    board_height: int = 11,
    board_width: int = 11,
    n_snakes: int = 2,
    lengths: tuple[int] = (3, 3),
    n_apples: int = 3,
    timeout: int = 500,
    turn: int = 16
) -> dict[str]:
    """Generate a random game state for testing purposes.
    /!\ The snakes can have less length than specified"""
    
    assert n_snakes == len(lengths), "The number of snakes must match the number of lengths"
    assert sum(lengths) + n_apples <= board_height * board_width, "The sum of the lengths and apples must fit in the board"
    
    snakes_matrix = []
    for _ in range(board_height):
        snakes_matrix.append([False for _ in range(board_width)])
    
    you = GenerateRandomSnake(snakes_matrix, "you", board_height, board_width, lengths[0])
    snakes = [you]
    for i in range(1, n_snakes):
        snakes.append(GenerateRandomSnake(snakes_matrix, f"snake_{i}", board_height, board_width, lengths[i]))
    
    return {
        "game": {"timeout": timeout},
        "turn": turn,
        "board": {
            "height": board_height,
            "width": board_width,
            "snakes": snakes,
            "food": [GetFreePosition(snakes_matrix, board_height, board_width) for _ in range(n_apples)],
        },
        "you": you,
    }