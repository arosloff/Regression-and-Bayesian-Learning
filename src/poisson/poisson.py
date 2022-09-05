import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    # *** START CODE HERE ***
    def plot(x, y, predictions, save_path):
        plt.figure()
        plt.scatter(y, predictions, c='coral')
        plt.xlabel('true count')
        plt.ylabel('predicted count')
        plt.savefig(save_path)

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    # Fit a Poisson Regression model
    poisson = PoissonRegression()
    poisson.fit(x_train, y_train)
    predictions = poisson.predict(x_valid)
    plot(x_valid, y_valid, predictions, '3d.png')
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=10e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])

        def updateTheta():
            for row in range(x.shape[0]):
                z = np.dot(np.transpose(self.theta), x[row])
                prediction = np.exp(z)
                error = y[row] - prediction
                k = 1e-5 * (error * x[row])
                self.theta = self.theta + k

        thetaDiff = np.inf
        iter = 0
        while True:
            if (thetaDiff < self.eps or iter > self.max_iter):
                break
            else:
                prevTheta = self.theta
                updateTheta()
                if (iter > 1):
                    thetaDiff = np.linalg.norm(self.theta - prevTheta, 1)
            iter += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = np.zeros(x.shape[0])
        for row in range(x.shape[0]):
            z = np.dot(np.transpose(self.theta), x[row])
            predictions[row] = np.exp(z)
        return predictions
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
