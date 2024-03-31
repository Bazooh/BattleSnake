from Game.cgame import *
from UCT.UCT import *
from UCT.Show import show_tree

snakes = [[(1, 10), (2, 10), (3, 10), (4, 10)], [(9, 5), (9, 4), (9, 3)]]
board = CBoard.from_board(Board(11, 11, snakes, []))

print(board)
tree = Tree.from_cboard(board)
UCT(tree, 0.1, Network())

board2 = next_board(tree.board, [0, 3])

print(board2.winner)
show_tree(tree)