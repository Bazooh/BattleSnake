from cgame import *
import unittest
from tqdm import tqdm

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class IntegerArithmeticTestCase(unittest.TestCase):
    def testMove(self) :
        snakes = [[(1, 10), (2, 10), (3, 10), (4, 10)], [(9, 5), (9, 4), (9, 3)]]
        board = CBoard.from_board(Board(11, 11, snakes, []))

        self.assertEqual(str(board), """\
. X X X X . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . X . 
. . . . . . . . . X . 
. . . . . . . . . X . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [DOWN, LEFT])
        
        self.assertEqual(str(board), """\
. X X X . . . . . . . 
. X . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . X X . 
. . . . . . . . . X . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [DOWN, LEFT])
        self.assertEqual(str(board), """\
. X X . . . . . . . . 
. X . . . . . . . . . 
. X . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . X X X . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [LEFT, UP])
        self.assertEqual(str(board), """\
. X . . . . . . . . . 
. X . . . . . . . . . 
X X . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . X . . . 
. . . . . . . X X . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [DOWN, RIGHT])
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. X . . . . . . . . . 
X X . . . . . . . . . 
X . . . . . . . . . . 
. . . . . . . X X . . 
. . . . . . . X . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [RIGHT, UP])
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. . . . . . . . . . . 
X X . . . . . . . . . 
X X . . . . . . X . . 
. . . . . . . X X . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [UP, RIGHT])
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. . . . . . . . . . . 
X X . . . . . . . . . 
X X . . . . . . X X . 
. . . . . . . . X . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [UP, RIGHT])
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. X . . . . . . . . . 
. X . . . . . . . . . 
X X . . . . . . X X X 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
    
    def testDieOutside(self) :
        snakes = [[(10, 3), (10, 2), (10, 1)], [(7, 8), (8, 8), (9, 8)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))

        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . . . . . X X X . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . X 
. . . . . . . . . . X 
. . . . . . . . . . X 
. . . . . . . . . . . 
""")
        board = next_board(board, [RIGHT, UP])
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, 1)
    
    def testDieColliding(self) :
        snakes = [[(4, 8), (5, 8), (6, 8)], [(5, 7), (5, 6), (5, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))
        
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . . X X X . . . . 
. . . . . X . . . . . 
. . . . . X . . . . . 
. . . . . X . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [LEFT, UP])
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, 0)
    
    def testDieColliding2(self) :
        snakes = [[(4, 8), (3, 8), (2, 8)], [(5, 7), (5, 6), (5, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))
        
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . X X X . . . . . . 
. . . . . X . . . . . 
. . . . . X . . . . . 
. . . . . X . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [RIGHT, UP])
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, NO_WINNER)

    def testDieColliding3(self) :
        snakes = [[(4, 8), (3, 8), (2, 8), (1, 8)], [(5, 7), (5, 6), (5, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))
        
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. X X X X . . . . . . 
. . . . . X . . . . . 
. . . . . X . . . . . 
. . . . . X . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [RIGHT, UP])
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, 0)  
    
    def testDieAtSameTime(self) :
        snakes = [[(4, 10), (4, 9), (4, 8), (3, 8)], [(0, 5), (1, 5), (2, 5)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))
        
        self.assertEqual(str(board), """\
. . . . X . . . . . . 
. O . . X . . . . . . 
. . . X X . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
X X X . . . . . . . . 
. . . . . . . . O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [UP, LEFT])
        self.assertTrue(board.finished)
        self.assertEqual(board.winner, NO_WINNER)  
    
    def testEat(self) :
        snakes = [[(4, 8), (5, 8), (6, 8)], [(7, 4), (6, 4), (5, 4)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))
        
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . . X X X . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . X X X O . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [LEFT, RIGHT])
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . . X X X . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . X X X . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        board = next_board(board, [LEFT, UP])
        self.assertEqual(str(board), """\
. . . . . . . . . . . 
. O . . . . . . . . . 
. . X X X . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . X . . 
. . . . . . X X X . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
. . . . . . . . . . . 
""")
        

    def testFailChrome(self) :
        snakes = [[(6, 0), (5, 0), (5, 1), (6, 1)], [(6, 2), (6, 3), (6, 4)]]
        board = CBoard.from_board(Board(11, 11, snakes, [(4, 10), (2, 0), (3, 3), (8, 0), (8, 1), (8, 3), (8, 4), (8, 7)]))
        
        print(board)
        
        board = next_board(board, [UP, DOWN])
        
        print(board.finished)
    
    def testCopy(self) :
        snakes = [[(4, 8), (5, 8), (6, 8)], [(7, 4), (6, 4), (5, 4)]]
        board1 = CBoard.from_board(Board(11, 11, snakes, [(1, 9), (8, 4)]))
        board2 = next_board(board1, [LEFT, RIGHT])
        
        self.assertNotEqual(str(board1), str(board2))
    
    def testLeak(self) :
        snakes = [[(0, 10), (1, 10), (1, 9), (0, 9)], [(9, 1), (10, 1), (10, 0), (9, 0)]]
        directions = [DOWN, RIGHT, UP, LEFT]
        board = CBoard.from_board(Board(11, 11, snakes, []))
        
        for i in tqdm(range(1_000_000)):
            board = next_board(board, [directions[i % 4], directions[i % 4]])


if __name__ == '__main__':
    unittest.main()