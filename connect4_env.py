import os
from copy import deepcopy

import numpy as np

PLAYER1 = 1
PLAYER2 = 2


def get_coord(col, row):
    """
    helper function to query the board matrix

    col: 0-6 inclusive
    row: 0-5 inclusive

    example usage: board.to_matrix()[get_coord(col=0, row=0)]
    useful in-case e.g. the layout of the matrix is changed later, converted to an array, etc
    """
    return col, row


class GameBoard:

    def __init__(self, board_array=None, next_player=PLAYER1):
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
        self.total_moves = 0
        # used to determine which player is next
        if next_player not in [PLAYER1, PLAYER2]:
            raise Exception("Invalid next_move_player argument")
        self.player1_goes_next = next_player == PLAYER1

        # for all 7 columns, the value marks the bit index where the next
        # fallen piece should go
        self.next_slot_index = [0, 7, 14, 21, 28, 35, 42]

        if board_array is not None:
            player1_move_count = 0
            for i in range(42):
                col = i // 6
                row = i % 6
                board_value = board_array[i]
                if board_value == 0:
                    continue  # the slot is not occupied
                self.total_moves += 1
                bit = 1 << (col * 7 + row)
                if board_value == 1:
                    player1_move_count += 1
                    self.player1_state |= bit
                self.player2_state_diff |= bit  # state diff bit is always set for occupied slots
            player2_move_count = self.total_moves-player1_move_count
            # check that valid next_move_player argument was passed
            # note that this check does not work if there is an equal amount of player1 & player2 moves
            if self.player1_goes_next and player1_move_count > player2_move_count\
                    or not self.player1_goes_next and player2_move_count > player1_move_count:
                raise Exception("Invalid next_move_player argument")

    def copy(self):
        return deepcopy(self)

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

    def get_next_player(self):
        return PLAYER1 if self.player1_goes_next else PLAYER2

    def make_move(self, col):
        if col < 0 or col >= 7:
            raise Exception("Invalid column: "+str(col))
        if not self.is_column_available(col):
            raise Exception("Column "+str(col)+" is unavailable")

        move = 1 << self.next_slot_index[col]
        self.player2_state_diff |= move  # always updated for any move
        if self.player1_goes_next:  # move of player1
            self.player1_state |= move

        self.next_slot_index[col] += 1
        self.total_moves += 1
        self.player1_goes_next = not self.player1_goes_next

        self.move_history.append(col)

    def undo_move(self):
        if len(self.move_history) == 0:
            return False
        col = self.move_history.pop()

        self.next_slot_index[col] -= 1
        self.total_moves -= 1
        self.player1_goes_next = not self.player1_goes_next

        move = 1 << self.next_slot_index[col]
        self.player2_state_diff ^= move  # always updated for any move
        if self.player1_goes_next:  # was move of player1
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

    def to_array(self, perspective=PLAYER1):
        array = np.zeros(42)
        me = 1 if perspective == PLAYER1 else 2
        opponent = 2 if perspective == PLAYER1 else 1
        for i in range(42):
            col = i // 6
            row = i % 6
            shift = col * 7 + row
            if (self._get_state(1) >> shift) & 1:
                array[i] = me
            elif (self._get_state(2) >> shift) & 1:
                array[i] = opponent
        return array

    def to_matrix(self):
        """
        dim1 is columns
        dim2 is rows
        matrix(7, 6)

        0,0 is the bottom-left corner
        6,5 is the top-right corner

        0 = unoccupied
        1 = player1
        2 = player2
        """
        return self.to_array().reshape((7, 6))

    def get_value_at(self, col, row):
        return self.to_matrix()[get_coord(col=col, row=row)]

    def print(self, cls=False):
        if cls:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(self)+"\n")

    def __str__(self):
        matrix = self.to_matrix()
        output = ""
        for row in range(matrix.shape[1]-1, -1, -1):  # top to bottom (5->0)
            if len(output) > 0:
                output += "\n"
            for col in range(matrix.shape[0]):  # left to right (0->6)
                val = matrix[col, row]
                output += "_" if val == 0 else "1" if val == 1 else "2"
        return output
