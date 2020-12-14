import utility as util
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main(train_path,  test_path, epochs):


    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)
    logreg = LogisticRegression(verbose=True, penalty='l2', C=3, tol=1e-7, fit_intercept=True, solver='newton-cg',
                               multi_class='multinomial', max_iter=epochs)

    logreg.fit(train_x, train_t)
    pred_lg_prob = logreg.predict(test_x)
    return accuracy_score(test_t, pred_lg_prob)

def cm_base(train_path,  test_path, epochs):


    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)
    logreg = LogisticRegression(verbose=True, penalty='l2', C=3, tol=1e-7, fit_intercept=True, solver='newton-cg',
                                multi_class='multinomial', max_iter=epochs)

    logreg.fit(train_x, train_t)
    pred_lg_prob = logreg.predict(test_x)
    return  pred_lg_prob, test_t







