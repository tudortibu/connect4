import connect4_env as c4
import utility as util
import numpy as np


def get_endgame_boards(csv_path, out_path, filter_key, label_col='class'):
    # Validate label_col argument
    allowed_label_cols = 'class'
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        h = csv_fh.readline().strip()
        headers = h.split(',')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    lst = []

    for line in data:
        b = line[:42].reshape((6,7))
        print(b)
        board = c4.Connect4(b)
        if board.is_win(filter_key):
            lst.append(True)
        else:
            lst.append(False)

    data_copy = data[lst]
    data_copy.astype(int)

    np.savetxt(out_path, data_copy, fmt='%i', delimiter=',', header=h, comments='')
    return

file = "Connect4_Data/All_Moves/connect-4-win.csv"
out_file = "Connect4_Data/All_Moves/connect-4-clean-won.csv"
key = 1
get_endgame_boards(file, out_file, key)

