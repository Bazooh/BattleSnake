import unittest

from tqdm import tqdm
from typing import Literal

from Game.cgame import Board, CBoard
from Constants import NO_WINNER

LOCAL_LEFT = 0
STRAIGTH = 1
LOCAL_RIGHT = 2

LocalAction = Literal[LOCAL_LEFT, STRAIGTH, LOCAL_RIGHT]


class IntegerArithmeticTestCase(unittest.TestCase):
    def testMove(self):
        snakes = [[(1, 10), (2, 10), (3, 10), (4, 10)], [(9, 5), (9, 4), (9, 3)]]
        board = CBoard.from_board(Board(11, 11, snakes))

        self.assertEqual(
            str(board),
            """\
. X * * * . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . X . 
. . . . . . . . . * . 
. . . . . . . . . * . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((LOCAL_LEFT, LOCAL_LEFT))

        self.assertEqual(
            str(board),
            """\
. * * * . . . . . . . 
. X . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . X * . 
. . . . . . . . . * . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((STRAIGTH, STRAIGTH))
        self.assertEqual(
            str(board),
            """\
. * * . . . . . . . . 
. * . . . . . . . . . 
. X . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . X * * . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((LOCAL_RIGHT, LOCAL_RIGHT))
        self.assertEqual(
            str(board),
            """\
. * . . . . . . . . . 
. * . . . . . . . . . 
X * . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . X . . . 
. . . . . . . * * . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, False, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((LOCAL_LEFT, LOCAL_RIGHT))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. * . . . . . . . . . 
* * . . . . . . . . . 
X . . . . . . . . . . 
. . . . . . . * X . . 
. . . . . . . * . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, False])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((LOCAL_LEFT, LOCAL_LEFT))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. . . . . . . . . . . 
* * . . . . . . . . . 
* X . . . . . . X . . 
. . . . . . . * * . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((LOCAL_LEFT, LOCAL_RIGHT))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. . . . . . . . . . . 
* X . . . . . . . . . 
* * . . . . . . * X . 
. . . . . . . . * . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((STRAIGTH, STRAIGTH))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. X . . . . . . . . . 
. * . . . . . . . . . 
* * . . . . . . * * X 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, False, True])

    def testDieOutside(self):
        snakes = [[(10, 3), (10, 2), (10, 1)], [(7, 8), (8, 8), (9, 8)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . . . . . X * * . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . X 
. . . . . . . . . . * 
. . . . . . . . . . * 
. . . . . . . . . . . 
""",
        )
        board = board.next((LOCAL_RIGHT, LOCAL_LEFT))
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, -1)

    def testDieColliding(self):
        snakes = [[(4, 8), (5, 8), (6, 8)], [(5, 7), (5, 6), (5, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . . X * * . . . . 
. . . . . X . . . . . 
. . . . . * . . . . . 
. . . . . * . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((LOCAL_LEFT, STRAIGTH))
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, 1)

    def testDieColliding2(self):
        snakes = [[(4, 8), (3, 8), (2, 8)], [(5, 7), (5, 6), (5, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . * * X . . . . . . 
. . . . . X . . . . . 
. . . . . * . . . . . 
. . . . . * . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((STRAIGTH, STRAIGTH))
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, NO_WINNER)

    def testDieColliding3(self):
        snakes = [[(4, 8), (3, 8), (2, 8), (1, 8)], [(5, 7), (5, 6), (5, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. * * * X . . . . . . 
. . . . . X . . . . . 
. . . . . * . . . . . 
. . . . . * . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((STRAIGTH, STRAIGTH))
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, 1)

    def testDieColliding4(self):
        snakes = [[(1, 9), (1, 8), (1, 7), (1, 6), (2, 6), (3, 6), (3, 5)], [(4, 6), (5, 6), (6, 6)]]
        board = CBoard.from_board(Board(11, 11, snakes))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. X . . . . . . . . . 
. * . . . . . . . . . 
. * . . . . . . . . . 
. * * * X * * . . . . 
. . . * . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )

        board = board.next((STRAIGTH, STRAIGTH))
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, 1)

    def testDieColliding5(self):
        snakes = [[(6, 3), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)], [(1, 9), (1, 8), (1, 7), (1, 6), (2, 6), (3, 6), (3, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(7, 3)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. X . . . . . . . . . 
. * . . . . . . . . . 
. * . . . . . . . . . 
. * * * . . . . . . . 
. . . * . . . . . . . 
. . . . . . . . * . . 
. . . . . . X O * . . 
. . . . . . * * * . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )

        board = board.next((LOCAL_RIGHT, STRAIGTH))

        self.assertEqual(
            str(board),
            """\
. X . . . . . . . . . 
. * . . . . . . . . . 
. * . . . . . . . . . 
. * . . . . . . . . . 
. * * * . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . * X * . . 
. . . . . . * * * . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )

        board = board.next((STRAIGTH, LOCAL_RIGHT))

        self.assertTrue(board.finished)
        self.assertEqual(board.winner, -1)

    def testDieAtSameTime(self):
        snakes = [[(4, 10), (4, 9), (4, 8), (3, 8)], [(0, 5), (1, 5), (2, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . X . . . . . . 
. O . . * . . . . . . 
. . . * * . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
X * * . . . . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((STRAIGTH, STRAIGTH))
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, NO_WINNER)

    def testWalkOnTail(self):
        snakes = [[(4, 8), (5, 8), (6, 8)], [(1, 8), (2, 8), (3, 8)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. X * * X * * . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((STRAIGTH, STRAIGTH))
        self.assertFalse(board.finished)

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
X * * X * * . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )

    def testEat(self):
        snakes = [[(4, 8), (5, 8), (6, 8)], [(7, 4), (6, 4), (5, 4)]]
        board = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))

        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . . X * * . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . * * X O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        board = board.next((STRAIGTH, STRAIGTH))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . X * * . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . * * X . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

        board = board.next((STRAIGTH, LOCAL_LEFT))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . X * * . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . X . . 
. . . . . . * * * . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

    def testPlayableActions(self):
        snakes = [[(6, 8), (5, 8), (4, 8)], [(7, 6), (7, 5), (7, 4)]]
        board = CBoard.from_board(Board(11, 11, snakes))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . * * X . . . . 
. . . . . . . . . . . 
. . . . . . . X . . . 
. . . . . . . * . . . 
. . . . . . . * . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )

        board = board.next((STRAIGTH, STRAIGTH))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . * * X . . . 
. . . . . . . X . . . 
. . . . . . . * . . . 
. . . . . . . * . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, False])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, False, True])

        board = board.next((STRAIGTH, LOCAL_LEFT))
        self.assertEqual(
            str(board),
            """\
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . * * X . . 
. . . . . . X * . . . 
. . . . . . . * . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""",
        )
        self.assertEqual(list(board.snakes[0].contents.playable_actions), [True, True, True])
        self.assertEqual(list(board.snakes[1].contents.playable_actions), [True, True, True])

    def testFailChrome(self):
        snakes = [[(6, 0), (5, 0), (5, 1), (6, 1)], [(6, 2), (6, 3), (6, 4)]]
        board = CBoard.from_board(
            Board(11, 11, snakes, apples=[(4, 10), (2, 0), (3, 3), (8, 0), (8, 1), (8, 3), (8, 4), (8, 7)])
        )

        board = board.next((LOCAL_LEFT, STRAIGTH))

        self.assertEqual(board.finished, True)
        self.assertEqual(board.winner, 1)

    def testCopy(self):
        snakes = [[(4, 8), (5, 8), (6, 8)], [(7, 4), (6, 4), (5, 4)]]
        board1 = CBoard.from_board(Board(11, 11, snakes, apples=[(1, 9), (8, 4)]))
        board2 = board1.next((STRAIGTH, STRAIGTH))

        self.assertNotEqual(str(board1), str(board2))

    def testLeak(self):
        snakes = [[(0, 10), (1, 10), (1, 9), (0, 9)], [(9, 1), (10, 1), (10, 0), (9, 0)]]
        board = CBoard.from_board(Board(11, 11, snakes))

        for _ in tqdm(range(100_000)):
            board = board.next((LOCAL_LEFT, LOCAL_LEFT))


if __name__ == "__main__":
    unittest.main()
