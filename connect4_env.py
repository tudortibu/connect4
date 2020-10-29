import os
import numpy as np

PLAYER1 = 1
PLAYER2 = 2


class GameBoard:

    def __init__(self, board_matrix=None):
        # low 49 used bits to represent the board state (42+7 for sentinel row)
        # we need the sentinel row for bitwise O(1) four-in-a-row checking
        # all it does is delimit each column in the mask with a zero bit
        # bitmask of player 1
        self.player1_state = 0
        # XOR difference bitmask of player 2
        self.player2_state_diff = 0

        # history of moves made following the current object instantiation
        self.move_history = []

        # keeps track of total moves made by both players
        # also used to determine which player is next
        self.total_moves = 0

        # for all 7 columns, the value marks the bit index where the next
        # fallen piece should go
        self.next_slot_index = [0, 7, 14, 21, 28, 35, 42]

        if board_matrix is not None:
            for col in range(7):
                for row in range(6):
                    bit = 1 << (col*7 + row)
                    board_value = board_matrix[row, col]
                    if board_value == 0:
                        continue  # the slot is not occupied
                    self.player2_state_diff |= bit  # state diff bit is always set for occupied slots
                    if board_value == 1:
                        self.player1_state |= bit

    def _get_state(self, player):
        return self.player1_state if player == 1 else self.player1_state ^ self.player2_state_diff

    def get_winner(self):
        if self.has_won(PLAYER1):
            return PLAYER1
        if self.has_won(PLAYER2):
            return PLAYER2
        return None

    # TODO has 2 in-a-row?
    # TODO has 3 in-a-row?

    def has_won(self, player):
        directions = [1, 6, 7, 8]
        for direction in directions:
            state = self._get_state(player)
            squished_state = state & (state >> direction)
            if (squished_state & (squished_state >> (2*direction))) != 0:
                return True
        return False

    def make_move(self, col):
        if col < 0 or col >= 7:
            raise Exception("Invalid column: "+str(col))
        if not self.is_column_available(col):
            raise Exception("Column "+str(col)+" is unavailable")

        move = 1 << self.next_slot_index[col]
        self.player2_state_diff |= move  # always updated for any move
        if self.total_moves % 2 == 0:  # move of player1
            self.player1_state |= move

        self.next_slot_index[col] += 1
        self.total_moves += 1

        self.move_history.append(col)

    def undo_move(self):
        if len(self.move_history) == 0:
            return False
        col = self.move_history.pop()

        self.next_slot_index[col] -= 1
        self.total_moves -= 1

        move = 1 << self.next_slot_index[col]
        self.player2_state_diff ^= move  # always updated for any move
        if self.total_moves % 2 == 0:  # was move of player1
            self.player1_state ^= move

    def get_column_height(self, col):
        return self.next_slot_index[col] % 7

    def is_column_available(self, col):
        return self.get_column_height(col) < 6

    def get_available_columns(self):
        cols = []
        TOP = 0b1000000100000010000001000000100000010000001000000
        for col in range(0, 7):
            if (TOP & (1 << self.next_slot_index[col])) == 0:
                cols.append(col)
        return cols

    def to_matrix(self):
        """
        value at 0,0 would be top-left corner
        value at 5,6 would be bottom-right corner

        0 = unoccupied
        1 = player1
        2 = player2
        """
        matrix = np.zeros((6, 7))

        for row in range(6):
            for col in range(7):
                shift = col*7 + (5-row)
                if (self._get_state(1) >> shift) & 1:
                    matrix[row, col] = 1
                elif (self._get_state(2) >> shift) & 1:
                    matrix[row, col] = 2

        return matrix

    def print(self, cls=False):
        if cls:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(self)+"\n")

    def __str__(self):
        matrix = self.to_matrix()
        output = ""
        for row in range(matrix.shape[0]):
            if len(output) > 0:
                output += "\n"
            for col in range(matrix.shape[1]):
                val = matrix[row, col]
                output += "_" if val == 0 else "1" if val == 1 else "2"
        return output
