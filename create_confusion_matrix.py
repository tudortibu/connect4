import tensorflow as tf
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_confusion_matrix(labels, predictions):
    matrix = tf.math.confusion_matrix(labels, predictions)

    return matrix

def change_label (y):
    d = {0:0, 1:1, -1:2}

    yy = np.zeros((y.shape[0]))

    for i in range (len(y)):
        yy[i] = d[y[i]]

    return yy

def crop_matrix (x):
    row_del = np.delete(x, 0, 0)
    col_del = np.delete(row_del, 0, 1)
    print(col_del)
    return col_del

def create_confusion_matrix():
    lst = ["CNN", "MLP"]

    for i in lst:

        pred_class =  np.loadtxt(f"Weights/{i}_Weights/shuffled/pred_class.csv")
        true_labels_2  = np.loadtxt( f"Weights/{i}_Weights/shuffled/true2_class.csv")
        tl = change_label(true_labels_2)


        matrix = get_confusion_matrix(tl, pred_class)
        np.savetxt(f"Weights/{i}_Weights/shuffled/matrix.csv", matrix)
        print(matrix)
    return

def create_plot():
    lst = ["CNN", "MLP"]

    for i in lst:

        m =  np.loadtxt(f"Weights/{i}_Weights/shuffled/matrix.csv")
        m = crop_matrix(m)
        class_names = [ "Win" , "Loss"]
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            new_title = f"{i} {title}"
            cm = m
            if(normalize):
                cm= m.astype('float') / m.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp = disp.plot(cmap="Blues")
            disp.ax_.set_title(new_title)
            plt.savefig(f"Weights/{i}_Weights/shuffled/{new_title}.png")
            plt.show()

            print(new_title)
            print(disp.confusion_matrix)

    return
