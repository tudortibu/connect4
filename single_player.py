from connect4_env import Connect4
import random as r


def main():
    exit_state = False
    while not exit_state:
        c4 = Connect4()
        endgame = False
        c4.print_grid_pretty()
        while not endgame:
            print()
            print("Player Move")
            print(f"Available Moves: {c4.list_moves()}" )
            valid_inputs = list(map(str, c4.list_moves()))
            valid_inputs += ['e']
            print(valid_inputs)
            invalid_input = True

            while invalid_input:
                val = input("Enter Move(e to exit):")
                if val in valid_inputs:

                    invalid_input = False
                    break
                print(f"{val} is not a valid input")

            if val == 'e':
                exit_state = True
                break

            c4.make_move( int(val))
            c4.print_grid_pretty()

            if game_end(c4, 1):
                endgame = True
                break

            opp_move = r.choice(c4.list_moves())
            print(f"CPU move: {opp_move}")
            c4.make_move(opp_move)
            c4.print_grid_pretty()
            if game_end(c4, 2):
                endgame = True
                break


def game_end(c=Connect4, player=1):
    if len(c.list_moves()) == 0:
        print("Draw")
        return True
    elif c.is_win(player):
        print(f"Player {player} won!")
        return True
    return False

main()
