import os

from connect4_env import GameBoard, PLAYER2

from dql.old.trainer import Agent, Model


def main():
    weights_storage_path = "weights.h5"
    model = Model()
    if os.path.isfile(weights_storage_path):
        model.load_weights(weights_storage_path)

    agent = Agent(model, 0)

    board = GameBoard()
    board.print()
    while True:
        valid_inputs = list(map(lambda x: str(x+1), board.get_available_columns()))
        valid_inputs += ['e']

        while True:
            val = input("Enter Move 1-7 (e to exit): ")
            if val in valid_inputs:
                break
            print(f"{val} is not a valid input")

        if val == 'e':
            break

        board.make_move(int(val)-1)
        board.print()
        if game_end(board, 1):
            break

        opp_move = agent.choose_move(board.to_array(perspective=PLAYER2), 0)
        print(f"Agent's move: {opp_move+1}")
        if not board.is_column_available(opp_move):
            print("The agent's move was invalid. You win!")
            break
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
