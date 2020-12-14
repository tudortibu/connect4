import os
import numpy as np


def get_data(folder):
    files = os.listdir(folder)
    data = np.zeros((len(files),19, 2))
    for i in range(len(files)):
        curr_data = np.loadtxt( (folder + "/" + files[i]), delimiter=',', skiprows=1)
        data[i] = curr_data
    return data

def get_best(data, fmt="avg"):
    if fmt=="avg" :
        avgs = np.mean(data,axis=1)
        best_avg = np.argmax(avgs, axis=0)
        return data[best_avg[1]]

    if fmt == "max":
        maxes = np.max(data, axis=1)
        max_data = np.argmax(maxes, axis=0)
        return data[max_data[1]]


file_names = ["mlp", "gbm", "logreg", "conv_nn", "svm"]
header = "Epochs, Accuracy"

for file in file_names:
    folder = f"Graphs/{file}/epoch_tests/Only_Incomplete_Games"
    data = get_data(folder)
    best_data_avg = get_best(data, "avg")
    best_data_max = get_best(data, "max")
    avg_out = folder + "/best_avg.csv"
    max_out = folder + "/best_max.csv"
    np.savetxt(avg_out, best_data_avg, delimiter=',', header=header, comments='')
    np.savetxt(max_out, best_data_max, delimiter=',', header=header, comments='')
