import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    logReg = LogisticRegression()
    # Train a logistic regression classifier
    logReg.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    y, probabilities = logReg.predict(x_train)
    util.plot(x_valid, y_valid, logReg.theta, '1b-ds2.png')

    # Use np.savetxt to save predictions on eval set to save_path
    # A bit confusing if you want raw prediction values or the final binary classification. If you want the latter pass `y` to this instead.
    np.savetxt(save_path, probabilities)
    # *** END CODE HERE ***


class LogisticRegression:
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
        self.theta = np.zeros(x.shape[1])

        # *** FIT HELPERS ***
        def sigmoid(row, theta):
            z = np.dot(np.transpose(theta), row)
            return (1/(1+(np.exp(-z))))
        def gradient(row, theta, y):
            return (sigmoid(row, theta) - y) * row
        def hessian(row, theta):
            sigmoidDerivative = sigmoid(row, theta) * (1 - sigmoid(row, theta))
            outerProduct = np.outer(row, row)
            return sigmoidDerivative * outerProduct
        def updateTheta():
            grad = np.zeros(x.shape[1])
            dims = (x.shape[1], x.shape[1])
            hess = np.zeros(dims)
            for row in range(x.shape[0]):
                rowGradient = gradient(x[row], self.theta, y[row])
                rowHessian = hessian(x[row], self.theta)
                grad += rowGradient
                hess += rowHessian
            grad /= x.shape[0]
            hess /= x.shape[0]
            self.theta = self.theta - np.dot(np.linalg.inv(hess), grad)

        # *** FITTING ***
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
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = np.zeros(x.shape[0])
        probabilities = np.zeros(x.shape[0])
        def sigmoid(row, theta):
            z = np.dot(np.transpose(theta), row)
            return (1/(1+(np.exp(-z))))

        for row in range(x.shape[0]):
            prob = sigmoid(x[row], self.theta)
            probabilities[row] = prob
            if (prob >= 0.5):
                predictions[row] = 1.0
        return predictions, probabilities
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
