import matplotlib.pyplot as plt
import fileinput
import numpy as np

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def load_dataset(csv_path, label_col='class', add_intercept=False):

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = 'class'
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')


    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i] != label_col]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def filter_x_data(data, filter):
    for line in range (len(data)):
        for val in range (len( data[line])):
            if data[line][val] == filter:
                data[line][val] = 0
    return data


def class_filtered_dataset(csv_path, out_path, label_col='class', filter_key=0):

    # Validate label_col argument
    allowed_label_cols = 'class'
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        h =csv_fh.readline().strip()
        headers = h.split(',')

    win_data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    lst= []

    for line in win_data:
        if line[42] == filter_key:
           lst.append(True)
        else:
            lst.append(False)

    win_data_copy = win_data[lst]
    win_data_copy.astype(int)


    np.savetxt(out_path, win_data_copy, fmt='%i', delimiter=',', header=h, comments='')
    return

def filter_all_data(csv_path, out_path, d=None, label_col='class'):

    # Validate label_col argument
    allowed_label_cols = 'class'
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))


    with open(csv_path, 'r') as csv_fh:
        h =csv_fh.readline().strip()
        #print(h)
        headers = h.split(',')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    print(data)
    data_copy = data.copy()
    if d is not None:
        for i in range( data.shape[0]):
            for j in range( data.shape[1]):
                data_copy[i][j] = str( d[data[i][j]])

    print(data_copy)
    np.savetxt(out_path, data_copy, fmt='%i', delimiter=',', header=h, comments='')
    return

def rotate_kaggle_data(csv_path, out_path, key, label_col='class'):
    # Validate label_col argument
    allowed_label_cols = 'class'
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))


    with open(csv_path, 'r') as csv_fh:
        h =csv_fh.readline().strip()
        #print(h)
        headers = h.split(',')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    print(data)
    data_copy = data.copy()

    for i in range(42):
        data_copy[i] = data[key[i]]

    np.savetxt(out_path, data_copy, fmt='%i', delimiter=',', header=h, comments='')
    return

"""
kaggle_key = { 0:35, 1:28, 2:21, 3:14, 4:7, 5:0,
               6:36, 7:29, 8:22, 9:15, 10:8, 11:1,
               12:37, 13:30, 14:23, 15:16, 16:9, 17:2,
               18:38, 19:31, 20:24, 21:17, 22:10, 23:3,
               24:39, 25:32, 26:25, 27:18, 28:11, 29:4,
               30:40, 31:33, 32:26, 33:19, 34:12, 35:5,
               36:41, 37:34, 38:27, 39:20, 40:13, 41:6
             }

file = "Connect4_Data/All_Moves/connect-4-clean.csv"
out_file = "Connect4_Data/All_Moves/connect-4-losing.csv"
class_filtered_dataset(file, out_file, filter_key=-1)
"""