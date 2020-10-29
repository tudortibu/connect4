import utility as util
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import warnings


def main(train_path,  test_path , valpath):

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    #train_t = change_label(train_t)
    #val_x, val_t = util.load_dataset(valpath, label_col='class')
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(42,), name="digits")
    x = tf.keras.layers.Dense(256, activation="sigmoid", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(256, activation="sigmoid", name="dense_2")(x)
    x = tf.keras.layers.Dense(64, activation="sigmoid", name="dense_3")(x)
    x = tf.keras.layers.Dense(64, activation="sigmoid", name="dense_4")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    warnings.filterwarnings('ignore')
    model.fit(train_x, train_t, validation_split=0.1, epochs=100)
    model.save("model")
    pred_nn_prob = model.predict(test_x, verbose=2)

    for i in range(len(test_t)):
        print(f"Predicted Decision Tree value: {pred_nn_prob[i]} True Value: {test_t[i]}")


def change_label (y):
    d = {0:[1,0,0], 1:[0,1,0], 2:[0,0,1]}

    yy = np.zeros((y.shape[0], 3))

    for i in range (len(y)):
        yy[i] = d[y[i]]

    return yy


main("Connect4_Data/8-ply Moves/connect-4-clean-train.csv","Connect4_Data/8-ply Moves/connect-4-clean-test.csv",
     "Connect4_Data/8-ply Moves/connect-4-clean-train-2.csv")

