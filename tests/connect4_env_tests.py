import random
import unittest
import numpy as np

from connect4_env import GameBoard


class TestGameBoard(unittest.TestCase):

    def test_random_game(self):
        board = GameBoard()
        while len(board.get_available_columns()) > 0 and board.get_winner() is None:
            board.make_move(random.choice(board.get_available_columns()))
            board.print()

    def test_connect_4a(self):
        board = GameBoard()
        board.make_move(0)
        board.make_move(1)
        board.make_move(0)
        board.make_move(1)
        board.make_move(0)
        board.make_move(1)
        board.make_move(0)
        self.assertTrue(board.get_winner() == 1)

    def test_undo(self):
        board = GameBoard()

        matrix = board.to_matrix()
        self.assertTrue(np.array_equal(matrix, np.zeros((6, 7))))

        board.make_move(2)
        board.make_move(5)
        board.make_move(2)
        board.make_move(3)
        matrix[5, 2] = 1
        matrix[5, 5] = 2
        matrix[4, 2] = 1
        matrix[5, 3] = 2
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

        board.undo_move()  # undo player2 move
        matrix[5, 3] = 0
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

        board.undo_move()  # undo player1 move
        matrix[4, 2] = 0
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

        board.make_move(0)  # player1 move again
        matrix[5, 0] = 1
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))
