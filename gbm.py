import utility as util
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def main(train_path,  test_path, epochs):

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)

    classifier = GradientBoostingClassifier(n_estimators=epochs)
    classifier.fit(train_x, train_t)

    pred_gbm_prob = classifier.predict(test_x)
    return accuracy_score(test_t, pred_gbm_prob)

def cm_base(train_path,  test_path, epochs):

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)

    classifier = GradientBoostingClassifier(n_estimators=epochs)
    classifier.fit(train_x, train_t)

    pred_gbm_prob = classifier.predict(test_x)
    return pred_gbm_prob, test_t