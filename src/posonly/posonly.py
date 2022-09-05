import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()

    x_test, t_test = util.load_dataset(test_path, 't', add_intercept=True)
    logReg1 = LogisticRegression()
    logReg1.fit(x_test, t_test)
    t, probabilities2a = logReg1.predict(x_test)
    util.plot(x_test, t, logReg1.theta, '2a.png')
    # A bit confusing if you want raw prediction values or the final binary classifications. If you want the latter pass `t` to this instead.
    np.savetxt(output_path_true, probabilities2a)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, 'y', add_intercept=True)
    logReg2 = LogisticRegression()
    logReg2.fit(x_train, y_train)
    y, probabilities2b = logReg2.predict(x_test)
    util.plot(x_test, t, logReg2.theta, '2b.png')
    # A bit confusing if you want raw prediction values or the final binary classifications. If you want the latter pass `y` to this instead.
    np.savetxt(output_path_naive, probabilities2b)

    # Part (f): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, 'y', add_intercept=True)
    logReg2 = LogisticRegression()
    logReg2.fit(x_train, y_train)
    v, probabilities2f = logReg2.predict(x_test)
    vSum = 0
    vCardinality = 0
    for i in range(y_valid.shape[0]-1):
        if y_valid[i] == 1:
            vSum += probabilities2f[i]
            vCardinality += 1
    alpha = vSum / vCardinality
    probabilities2f = probabilities2f * (1/alpha)
    util.plot(x_test, t, logReg2.theta, '2f.png', alpha)
    v = probabilities2f
    for i in range(v.shape[0]):
        if v[i] > 0.5:
            v[i] = 1
        else:
            v[i] = 0
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # A bit confusing if you want raw prediction values or the final binary classifications. If you want the latter pass `v` to this instead.
    np.savetxt(output_path_adjusted, probabilities2f)
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
