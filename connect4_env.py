import random as r
import numpy as np


class Connect4(object):

    def __init__(self, board=None, player=1):
        self.boards = {1:0, 2:0}
        self.history = []
        if board is None:
            self.counter = 0
            self.height = [0,7,14,21,28,35,42]
        else:
            self.get_position_mask_bitmap(board, player)

    def is_win(self, player=1):
        directions = [1,7,6,8]
        for direction in directions:
            bb = self.boards[player] & (self.boards[player] >> direction)
            if (bb & (bb >> (2*direction))) != 0:
                return True
        return False

    def make_move(self, col):
        move = 1 << self.height[col]
        self.height[col] += 1
        self.boards[(self.counter & 1) +1] ^= move
        self.counter += 1
        self.history.append(move)

    def undo_move(self):
        self.counter -= 1
        col = self.history[self.counter]
        self.height[col] -= 1
        move = 1 << self.height[col]
        self.boards[(self.counter & 1) + 1] ^= move

    def list_moves(self):
        moves = []
        TOP = 283691315109952
        for col in range(0,7):
            if (TOP & (1 << self.height[col])) == 0:
                moves.append(col)
        return moves

    def get_position_mask_bitmap(self, board, player):
        position, mask = '', ''
        cols = [0, 0, 0, 0, 0, 0, 0]

        for j in range(6, -1, -1):

            mask += '0'
            position += '0'
            temp = []

            for i in range(0, 6):
                mask += ['0', '1'][board[i, j] != 0]
                position += ['0', '1'][board[i, j] == player]
                if board[i,j] == 0:
                    temp.append(j)
            cols[i] = min(temp) + 7**i

        self.boards[0] = int(position, 2)
        self.boards[1] = self.boards[0] ^ int(mask, 2)
        self.height = cols
        self.counter = (bin(int(mask, 2)).count("1")) % 2

    def bitmap_to_array(self):
        p1_board = bin(self.boards[1])
        p2_board = bin(self.boards[2])

        p1_board = p1_board[2:]
        p2_board = p2_board[2:]

        p1_fill = ([0] * 49 + list(p1_board))[-49:]
        p1_fill.reverse()

        p2_fill = ([0] * 49 + list(p2_board))[-49:]
        p2_fill.reverse()

        board = np.zeros(49)

        for i in range(len( p1_fill)):
            if p1_fill[i] == '1':
                board[i] = 1

        for i in range(len( p2_fill)):
            if p2_fill[i] == '1':
                board[i] = 2
        return board

    def print_grid(self):
        board = self.bitmap_to_array()

        for i in range(5,-1, -1):
            string = ''
            for j in range(0, 7):
                val = j*7 + i

                string += str(board[val]) + " "
            print(string)

    def print_grid_pretty(self):
        board = self.bitmap_to_array()

        for i in range(5,-1, -1):
            string = ''
            for j in range(0, 7):
                val = j*7 + i

                if board[val] == 1 :
                    string += "o" + " "
                elif board[val] == 2 :
                    string += "x" + " "
                else:
                    string += "_" + " "
            print(string)



def main():
    c = Connect4()
    while len(c.list_moves()) > 0 :
        s = r.choice(c.list_moves())
        c.make_move(s)
        c.print_grid_pretty()
        print()

        """
        print("Move #: " + str( c.counter))
        print("Move : " + str(s))
        print("P1 board: " + str(c.boards[1]))
        print("P2 board: " + str(c.boards[2]))
        print()
        """
