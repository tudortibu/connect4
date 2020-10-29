import utility as util
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

def main(train_path,  test_path):

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)

    classifier = GradientBoostingClassifier()
    classifier.fit(train_x, train_t)


    pred_gbm_prob = classifier.predict(test_x)

    for i in range(len(test_t)):
        print(f"Predicted svm value: {pred_gbm_prob[i]} True Value: {test_t[i]}")
main("Connect4_Data/8-ply Moves/connect-4-clean-train.csv", "Connect4_Data/8-ply Moves/connect-4-clean-test.csv")