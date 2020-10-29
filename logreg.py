import utility as util
import numpy as np
from sklearn.linear_model import LogisticRegression

import sys

### NOTE : You need to complete logreg implementation first!
class LogisticRegression_from_class:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d, dtype=np.float32)

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)

            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)

            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(i, loss))

            if np.max(np.abs(prev_theta - self.theta)) < self.eps:
                break

        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))


        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = self._sigmoid(x.dot(self.theta))

        return y_hat

    def _gradient(self, x, y):
        """Get gradient of J.

        Returns:
            grad: The gradient of J with respect to theta. Same shape as theta.
        """
        n, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        grad = 1 / n * x.T.dot(probs - y)

        return grad

    def _hessian(self, x):
        """Get the Hessian of J given theta and x.

        Returns:
            hess: The Hessian of J. Shape (dim, dim), where dim is dimension of theta.
        """
        n, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        diag = np.diag(probs * (1. - probs))
        hess = 1 / n * x.T.dot(diag).dot(x)

        return hess

    def _loss(self, x, y):
        """Get the empirical loss for logistic regression."""
        eps = 1e-10
        hx = self._sigmoid(x.dot(self.theta))
        loss = -np.mean(y * np.log(hx + eps) + (1 - y) * np.log(1 - hx + eps))

        return loss

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # *** END CODE HERE ***


def main(train_path,  test_path):


    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)
    #print(test_x_inter)
    #classifier_t = LogisticRegression_from_class(max_iter=100, step_size=0.0001)
    #classifier_t.fit(train_x_inter, train_t)

    #pred_t_prob = classifier_t.predict(test_x_inter)
    logreg = LogisticRegression(verbose=True, penalty='l2', C=3, tol=1e-7, fit_intercept=True, solver='newton-cg',
                               multi_class='multinomial', max_iter=1000000000)

    logreg.fit(train_x, train_t)
    pred_lg_prob = logreg.predict(test_x)


    for i in range(len(test_t)):
        print(f"Predicted logreg value: {pred_lg_prob[i]} True Value: {test_t[i]}")

main("Connect4_Data/8-ply Moves/connect-4-clean-train.csv", "Connect4_Data/8-ply Moves/connect-4-clean-test.csv")









