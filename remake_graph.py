import matplotlib.pyplot as plot
import os
import numpy as np
import utility


graphs =[
    "Graphs/mlp/epoch_tests/Complete_Games_Data/2LFC_Dropout.csv",
    "Graphs/mlp/epoch_tests/Complete_Games_Data/3LFC1FC_Dropout.csv",
    "Graphs/mlp/epoch_tests/Complete_Games_Data/3LFC_Dropout.csv",
    "Graphs/mlp/epoch_tests/Complete_Games_Data/5SFC_Dropout.csv",
    "Graphs/mlp/epoch_tests/Complete_Games_Data/6FC_2Dropout.csv",
    "Graphs/mlp/epoch_tests/Complete_Games_Data/6FC_2Dropout_2.csv",
    "Graphs/mlp/epoch_tests/Complete_Games_Data/5128FC_Dropout.csv"
]
names = [
"Network 1",
"Network 2",
"Network 3",
"Network 4",
"Network 5",
"Network 6",
"Network 7",
]

data = []
for i in range (len(graphs)):
    data_file = np.loadtxt(graphs[i], delimiter=',', skiprows=1)
    data.append(data_file)


def create_graph(data, save_path, names):
    counter = 0
    for data_points in data :
        print(data_points)
        print(data_points[0])
        plot.plot(data_points[:,0], data_points[:,1], label=names[counter] )
        counter+=1
    plot.legend()
    plot.ylim(0.7, 1)
    plot.title("Accuracies of Different MLP Configurations")
    plot.xlabel("Epochs")
    plot.ylabel("Accuracy")
    plot.savefig(save_path)
    return

create_graph(data, "Graphs/mlp/epoch_tests/graph.png", names )