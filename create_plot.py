import numpy as np
import matplotlib.pyplot as plot
import os
def get_data(folder):
    files = os.listdir(folder)
    data = np.zeros((len(files),19, 2))
    for i in range(len(files)):
        curr_data = np.loadtxt( (folder + "/" + files[i]), delimiter=',', skiprows=1)
        data[i] = curr_data
    return data, files

def create_graph(data, save_path, names):
    counter = 0
    for data_points in data :
        print(data_points)
        print(data_points[0])
        plot.plot(data_points[:,0], data_points[:,1], label=names[counter] )
        counter+=1
    plot.legend()
    plot.ylim(0.3, 1)
    plot.title("All Classifiers, Train Dataset 2, Test on Dataset 1, Without Draws")
    plot.xlabel("Epochs")
    plot.ylabel("Accuracy")
    plot.savefig(save_path)
    return

def main(folder, save_path):
    data, names = get_data(folder)
    create_graph(data, save_path, names)

def seperate_folders_data(names, test="epoch_tests", folder="Data", file="best_avg"):
    data = np.zeros((len(names),19, 2))
    i = 0
    for name in names:
        file_loc = f"Graphs/{name}/{test}/{folder}/{file}.csv"
        curr_data = np.loadtxt( file_loc, delimiter=',', skiprows=1)
        data[i] = curr_data
        i+=1
    return data

def create_plot(save_path, names):
    data = seperate_folders_data(names, folder="train_Complete_test_Incomplete_No_Draws", file="data")
    create_graph(data, save_path, names)

out_path = "Graphs/For Paper/all_train_Complete_test_Incmplete_No_Draws.png"
names = ["svm", "gbm", "logreg", "conv_nn", "mlp"]
create_plot(out_path, names)

#main("Graphs/mlp/epoch_tests/Data", "Graphs/mlp/epoch_tests/graph.png")