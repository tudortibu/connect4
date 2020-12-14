import numpy as np
import gbm, svm, logreg
from create_confusion_matrix import get_confusion_matrix, crop_matrix, change_label
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plot

train_file = "Connect4_Data/train_data/Incomplete_No_Draws/train-data.csv"
test_files = "Connect4_Data/train_data/Finished_Games_Draws/test-data.csv"


d = {"svm":svm, "gbm":gbm, "logreg":logreg}

def create_data(train_file, test_files, classifier, out_path ):
    epoch_range = range(1,20)
    data = np.zeros( (len(epoch_range),2) )
    header = "Epochs, Accuracy"


    for i in epoch_range:
        epochs = i*25
        data[i-1][0] = epochs

        data[i-1][1] = classifier.main(train_file, test_files, epochs=epochs)

    np.savetxt(out_path, data, delimiter=',', header=header, comments='')



def epoch_test():
    for classifier in d.keys() :
        out_path = f"Graphs/{classifier}/epoch_tests/train_Incomplete_test_Complete_No_Draws/data.csv"
        create_data(train_file, test_files, d[classifier], out_path)

def create_confusion_matrix():
    epochs = 275
    for classifier in d.keys() :
        out_path = f"Graphs/{classifier}/epoch_tests/train_Incomplete_test_Complete_No_Draws/confusion_matrix.csv"
        pred, labels = d[classifier].cm_base(train_file, test_files, epochs)
        pred = change_label(pred)
        labels = change_label(labels)
        cm = get_confusion_matrix(labels, pred)
        np.savetxt(out_path, cm)
    return


def create_cm_plot(file_in, file_out, classifier, crop=False):
        m =  np.loadtxt(file_in)
        class_names = ["Draw","Win" , "Loss"]
        if(crop):
            m = crop_matrix(m)
            class_names = ["Win" , "Loss"]

        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            new_title = f"{classifier} {title}"
            cm = m
            if(normalize):
                cm= m.astype('float') / m.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp = disp.plot(cmap="Blues")
            disp.ax_.set_title(new_title)
            plot.savefig(f"{file_out}/{new_title}.png")


            print(new_title)
            print(disp.confusion_matrix)
        return

def create_cm_plots():
    create_confusion_matrix()
    for classifier in d.keys() :
        in_path = f"Graphs/{classifier}/epoch_tests/train_Incomplete_test_Complete_No_Draws/confusion_matrix.csv"
        out_path = f"Graphs/{classifier}/epoch_tests/train_Incomplete_test_Complete_No_Draws"
        create_cm_plot(in_path, out_path, classifier, crop=True)


epoch_test()
create_cm_plots()
