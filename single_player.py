from connect4_env import GameBoard
import random as r


def main():
    board = GameBoard()
    board.print()
    while True:
        print()
        print("Player Move")
        print(f"Available Moves: {board.get_available_columns()}")
        valid_inputs = list(map(str, board.get_available_columns()))
        valid_inputs += ['e']
        print(valid_inputs)

        while True:
            val = input("Enter Move (e to exit): ")
            if val in valid_inputs:
                break
            print(f"{val} is not a valid input")

        if val == 'e':
            break

        board.make_move(int(val))
        board.print()
        if game_end(board, 1):
            break

        opp_move = r.choice(board.get_available_columns())
        print(f"CPU move: {opp_move}")
        board.make_move(opp_move)
        board.print()
        if game_end(board, 2):
            break


def game_end(board, player=1):
    if len(board.get_available_columns()) == 0:
        print("Draw")
        return True
    elif board.has_won(player):
        print(f"Player {player} won!")
        return True
    return False


main()
