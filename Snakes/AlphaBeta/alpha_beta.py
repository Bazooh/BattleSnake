from Constants import LOCAL_ACTIONS, INF, NO_WINNER, GameState, LocalAction, Move
from Game.cgame import CBoard
from Snakes.Snake import Snake


class AlphaBetaSnake(Snake):
    def move(self, game_state: GameState) -> Move:
        if len(game_state["board"]["snakes"]) < 2:
            return {"move": "down"}

        board = CBoard.from_game_state(game_state)

        value, action = self.alpha_beta_search(board, 8)

        assert action is not None, "No action found"

        return {"move": board.local_action_to_global(action, 0)}

    def evaluate(self, board: CBoard) -> float:
        if board.finished:
            if board.winner == NO_WINNER:
                return -10
            return int(board.winner) * 1000

        main_score = len(board.snakes[0].contents)
        other_score = len(board.snakes[1].contents)

        return main_score - other_score

    def alpha_beta_search(
        self,
        board: CBoard,
        depth: int,
        is_maximizing: bool = True,
        alpha: float = -INF,
        beta: float = INF,
        main_action: LocalAction | None = None,
    ) -> tuple[float, LocalAction | None]:
        if board.finished or depth == 0:
            return self.evaluate(board), None

        best_value = -INF if is_maximizing else INF
        best_action = None

        for local_action in LOCAL_ACTIONS:
            if main_action is None:
                new_board = board
            else:
                new_board = board.next((main_action, local_action))

            value, _ = self.alpha_beta_search(
                new_board, depth - 1, not is_maximizing, alpha, beta, local_action if main_action is None else None
            )

            if is_maximizing:
                if value > best_value:
                    best_value = value
                    best_action = local_action

                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_action = local_action

                beta = min(beta, best_value)

            if beta <= alpha:
                break

        return best_value, best_action
