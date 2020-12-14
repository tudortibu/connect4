import neural_net as nn
import mlp
import numpy as np


train_file = "Connect4_Data/train_data/Finished_Games_No_Draws/train-data.csv"
test_files = "Connect4_Data/train_data/No_Finished_Games/test-data.csv"
out_path_mlp = "Graphs/mlp/epoch_tests/train_Complete_test_Incomplete_No_Draws/data.csv"
out_path_conv = "Graphs/conv_nn/epoch_tests/train_Complete_test_Incomplete_No_Draws/data.csv"
epoch_range = range(1,20)
data_mlp = np.zeros( (len(epoch_range),2) )
data_conv = np.zeros( (len(epoch_range),2) )
header = "Epochs, Accuracy"
"""

for i in epoch_range:
    epochs = i*25
    data_conv[i-1][0] = epochs
    data_mlp[i-1][0] = epochs
    data_conv[i-1][1] = nn.main(train_file, test_files, epoch=epochs)
    data_mlp[i-1][1] = mlp.main(train_file, test_files, epoch=epochs)

np.savetxt(out_path_conv, data_conv, delimiter=',', header=header, comments='')
np.savetxt(out_path_mlp, data_mlp, delimiter=',', header=header, comments='')
"""
nn.main(train_file, test_files, epoch=275, save_weights=True)
mlp.main(train_file, test_files, epoch=275, save_weights=True)