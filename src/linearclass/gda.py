import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    gda = GDA()
    # Train a GDA classifier
    gda.fit(x_train, y_train)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    predictions, probabilities = gda.predict(x_valid)
    # Plot decision boundary on validation set
    util.plot(x_valid, y_valid, gda.theta, '1e-ds2.png')
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, probabilities)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        phi = 0
        for i in range(y.shape[0]):
            if y[i] == 1:
                phi += 1
        phi /= y.shape[0]

        mu_0_numerator = np.zeros(x.shape[1])
        mu_0_denominator = 0
        for row in range(x.shape[0]):
            if (y[row] == 0):
                mu_0_numerator += x[row]
                mu_0_denominator += 1
        mu_0 = mu_0_numerator / mu_0_denominator

        mu_1_numerator = np.zeros(x.shape[1])
        mu_1_denominator = 0
        for row in range(x.shape[0]):
            if (y[row] == 1):
                mu_1_numerator += x[row]
                mu_1_denominator += 1
        mu_1 = mu_1_numerator / mu_1_denominator

        sigma = np.zeros([x.shape[1], x.shape[1]])
        for row in range(x.shape[0]):
            if y[row] == 1:
                sigma += np.outer((x[row] - mu_1), x[row] - mu_1)
            else:
                sigma += np.outer((x[row] - mu_0), x[row] - mu_0)
        sigma /= x.shape[0]

        # Write theta in terms of the parameters
        self.theta = np.zeros(x.shape[1])

        theta_1 = np.dot(np.transpose(mu_0), np.linalg.inv(sigma)) - np.dot(np.transpose(mu_1), np.linalg.inv(sigma))

        # Theta 0 Calculations #
        sigmaDotMu1 = np.dot(np.linalg.inv(sigma), mu_1)
        sigmaDotMu0 = np.dot(np.linalg.inv(sigma), mu_0)

        scalarProduct1 = np.dot(np.transpose(mu_1), sigmaDotMu1)
        scalarProduct0 = np.dot(np.transpose(mu_0), sigmaDotMu0)

        scalarDifference = scalarProduct1 - scalarProduct0

        phiLog = np.log((1 - phi) / phi)

        theta_0 = (scalarDifference / 2) + phiLog

        self.theta = np.append(theta_0, theta_1)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = np.zeros(x.shape[0])
        probabilities = np.zeros(x.shape[0])

        for row in range(x.shape[0]):
            z = np.dot(np.transpose(self.theta), x[row])
            exp = np.exp((-1 * z))
            probabilities[row] = 1 / (1 + exp)
            if (probabilities[row] >= 0.5):
                predictions[row] = 1

        return predictions, probabilities
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
