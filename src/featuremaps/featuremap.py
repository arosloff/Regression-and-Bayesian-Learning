import numpy

import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        X_transpose = np.transpose(X)
        theta = np.linalg.solve(np.dot(X_transpose, X), (np.dot(X_transpose, y)))
        self.theta = theta
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        if (k == 0):
            return numpy.ones([X.shape[0],1])
        if (k == 1):
            return X
        else:
            new_x = np.zeros((X.shape[0], k+1), dtype=X.dtype)
            kRange = range(k+1)
            for row in range(new_x.shape[0]):
                kMap = map(lambda x: X[row][1]**x, kRange)
                new_x[row] =list(kMap)
        return new_x
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        if (k == 0):
            new_x = numpy.ones([X.shape[0],2])
            for row in range(new_x.shape[0]):
                new_x[row][1] = np.sin(X[row][1])
            return new_x
        else:
            new_x = np.zeros((X.shape[0], k+2), dtype=X.dtype)
            kRange = range(k+1)
            for row in range(new_x.shape[0]):
                kMap = map(lambda x: X[row][1]**x, kRange)
                sine = np.sin(X[row][1])
                new_x[row] = np.append(list(kMap), sine)
            return new_x
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(X, self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([70, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 70)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y, c='black')

    for k in ks:
        print(k)
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lin = LinearModel()
        if sine:
            X = lin.create_sin(k, train_x)
            new_plot_x = lin.create_sin(k, plot_x)
        else:
            X = lin.create_poly(k, train_x)
            new_plot_x = lin.create_poly(k, plot_x)

        lin.fit(X, train_y)
        plot_y = np.transpose(lin.predict(new_plot_x))
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(new_plot_x[:, 1], plot_y, label='k=%d' % k)
    plt.title(filename)
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    np.set_printoptions(suppress=True)
    run_exp(train_path, ks=[3], filename="5b-plot")
    run_exp(train_path, ks=[3, 5, 10, 20], filename="5c-plot")
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename="5d-plot")
    run_exp(small_path, ks=[1, 2, 5, 10, 20], filename = "5e-plot")
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
