import utility as util
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import datetime
import matplotlib.pyplot as plot
import warnings


def main(train_path,  test_path, epoch=15, valpath=None, save_weights=False):
    np.random.seed(0)
    tf.random.set_seed(0)

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    train_x, train_t = shuffle(train_x, train_t)
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    test_x, test_t = shuffle( test_x, test_t)
    train_t = change_label(train_t)
    test_tt = change_label(test_t)

    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential([
                                        tf.keras.layers.Reshape(( 6, 7, 1), input_shape=(42,1)),
                                        tf.keras.layers.Conv2D(256,5,
                                                               input_shape=(1, 6, 7, 1)),
                                        tf.keras.layers.Flatten(),
                                        #tf.keras.layers.Dropout(0.1),
                                        tf.keras.layers.Dense(450, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(210, activation=tf.nn.relu),
                                        #tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Dense(100, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.1),
                                        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
                                        ])

    warnings.filterwarnings('ignore')
    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    model.fit(train_x, train_t, validation_split=0.1, epochs=epoch, verbose=0)


    #model.save("model")
    pred_nn_prob = model.predict(test_x, verbose=0 )
    pred_nn_class = np.argmax(pred_nn_prob, axis=-1)
    acc = calculate_acc(pred_nn_class, test_tt, False)
    """
    for i in range(len(test_tt)):
        print(f"Predicted NN value: {pred_nn_prob[i]} True Value: {test_tt[i]}")
   
    if (save_weights):
        model.save("Weights/CNN_Weights/shuffled/model")
        np.savetxt("Weights/CNN_Weights/shuffled/pred_pb.csv",pred_nn_prob)
        np.savetxt("Weights/CNN_Weights/shuffled/pred_class.csv",pred_nn_class)
        np.savetxt("Weights/CNN_Weights/shuffled/true_class.csv", test_tt)
        np.savetxt("Weights/CNN_Weights/shuffled/true2_class.csv", test_t)
     """
    return #acc

def change_label (y):
    d = {0:[1,0,0], 1:[0,1,0], -1:[0,0,1]}

    yy = np.zeros((y.shape[0], 3))

    for i in range (len(y)):
        yy[i] = d[y[i]]

    return yy


def calculate_acc(results, labels, verbose=True):
    label_list = set (tuple(tuple(a_m.tolist()) for a_m in labels ))
    d = {}
    dd = {}
    #results_rounded = np.rint(results)
    #(label_list)
    for i in label_list:
        d[tuple(i)] = 0
        dd[tuple(i)] = 0
    for i in range(len(labels)):
        if verbose:
            print(f"Predicted NN value: {results[i]} True Value: {labels[i]}")
        d[tuple(labels[i].tolist())] += 1
        if labels[i][results[i]] == 1:
            dd[tuple( labels[i].tolist())] += 1
    total = 0
    t_correct = 0
    for i in label_list:
        total += d[i]
        t_correct += dd[i]
        print(f"Label {i} accuracy: {dd[i]/d[i] *100}%")
    print(f"Total accuracy: {t_correct/total *100}%")
    return t_correct/total


main