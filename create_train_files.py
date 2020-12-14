import utility as util
import numpy as np
from datetime import datetime
import os


def main(files, ratios=None, train_out=5000, out_folder="Connect4_Data/train_data", test_out=0):
    sub = create_sub_folder(out_folder)
    train, test = get_all_data(files, ratios, train_out, test_out)
    h = get_header(files[0])
    train_out_path = f"{sub}/train-data.csv"
    test_out_path = f"{sub}/test-data.csv"
    np.savetxt(train_out_path, train, fmt='%i', delimiter=',', header=h, comments='')
    np.savetxt(test_out_path, test, fmt='%i', delimiter=',', header=h, comments='')
    return

def create_sub_folder(out_folder):
    subfolder = datetime.now().strftime("%m-%d-%H-%M")
    sub = os.getcwd() + "/" + out_folder +"/" + subfolder
    subfolder = out_folder +"/" + subfolder
    os.makedirs(sub)
    return subfolder

def get_all_data(files, ratios, total_out, test_out):
    train_data = np.empty((0, 43), int)
    test_data = np.empty((0, 43), int)
    if ratios is not None:
        for file, ratio in zip(files,ratios):
            data = np.loadtxt(file, delimiter=',', skiprows=1)
            num_samples = int((total_out+test_out) * ratio)
            samples = data[np.random.choice(data.shape[0], size=num_samples, replace=False), :]
            index = int(total_out * ratio)
            training, test = samples[:index], samples[index:]
            train_data = np.concatenate((train_data, training))
            test_data = np.concatenate((test_data, test))
    else:
        all_data =  np.empty((0, 43), int)
        for file in files:
            data = np.loadtxt(file, delimiter=',', skiprows=1)
            all_data = np.concatenate(all_data, data)
        np.random.shuffle(all_data)
        test_data, train_data = all_data[:test_out] , all_data[test_out:]
    return train_data, test_data

def get_header(file):
    with open(file, 'r') as csv_fh:
        h =csv_fh.readline().strip()
    return h


def generate_data():
    files = [
            #"Connect4_Data/All_Moves/connect-4-lost.csv",
            #"Connect4_Data/All_Moves/connect-4-won.csv",
            #"Connect4_Data/All_Moves/connect-4-draw.csv",
            "Connect4_Data/8-ply Moves/connect-4-losing.csv",
            "Connect4_Data/8-ply Moves/connect-4-winning.csv"

             ]
    ratios =[1/2, 1/2]
    main(files, ratios, train_out=10000, test_out=5000)

generate_data()