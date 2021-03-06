import random
import unittest
import numpy as np

from connect4_env import GameBoard, get_coord


class TestGameBoard(unittest.TestCase):

    def test_random_game(self):
        board = GameBoard()
        while len(board.get_available_columns()) > 0 and board.get_winner() is None:
            board.make_move(random.choice(board.get_available_columns()))
            board.print()

    def test_init(self):
        board_matrix = np.zeros((7, 6))
        board_matrix[get_coord(col=0, row=0)] = 1
        board_matrix[get_coord(col=1, row=0)] = 2
        board_matrix[get_coord(col=0, row=1)] = 1
        board_matrix[get_coord(col=1, row=1)] = 2
        board_matrix[get_coord(col=0, row=2)] = 1
        board_matrix[get_coord(col=0, row=3)] = 2
        board_matrix[get_coord(col=1, row=2)] = 2

        board = GameBoard(board_matrix.flatten())
        self.assertTrue(np.array_equal(board.to_matrix(), board_matrix))

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
        self.assertTrue(np.array_equal(matrix, np.zeros((7, 6))))

        board.make_move(2)
        board.make_move(5)
        board.make_move(2)
        board.make_move(3)
        matrix[get_coord(col=2, row=0)] = 1
        matrix[get_coord(col=5, row=0)] = 2
        matrix[get_coord(col=2, row=1)] = 1
        matrix[get_coord(col=3, row=0)] = 2
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

        board.undo_move()  # undo player2 move
        matrix[get_coord(col=3, row=0)] = 0
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

        board.undo_move()  # undo player1 move
        matrix[get_coord(col=2, row=1)] = 0
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

        board.make_move(0)  # player1 move again
        matrix[get_coord(col=0, row=0)] = 1
        self.assertTrue(np.array_equal(matrix, board.to_matrix()))

    def test_copy(self):
        board = GameBoard()
        board.make_move(0)
        board.make_move(0)
        board.make_move(0)
        board.make_move(0)

        copy = board.copy()
        copy.undo_move()
        copy.undo_move()

        self.assertNotEqual(board.total_moves, copy.total_moves)
        self.assertNotEqual(board.next_slot_index, copy.next_slot_index)
