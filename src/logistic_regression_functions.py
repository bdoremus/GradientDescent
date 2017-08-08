import numpy as np
import unittest


def hyp(X):
    """Calculate the value of the Logistic (sigmoidal) hypothesis function;
    This is the basis of our classification model.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The sum of linear coefficients times x_values

    Returns
    -------
    The value of the sigmoid at the given point; a float between 0 and 1.
    """
    return 1./(1+np.exp(X))


def predict_proba(X, coeffs):
    """Calculate the predicted conditional probabilities (floats between 0 and
    1) for the given data with the given coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.

    Returns
    -------
    predicted_probabilities: The conditional probabilities from the logistic
        hypothosis function given the data and coefficients.

    """
    linear_predictor = -1*np.dot(X, coeffs)
    return hyp(linear_predictor)


def predict(X, coeffs, thres=0.5):
    """
    Calculate the predicted class values (0 or 1) for the given data with the
    given coefficients by comparing the predicted probabilities to a given
    threshold. almost)

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.
    threshold: Threshold for comparison of probabilities.

    Returns
    -------
    predicted_class: The predicted class.
    """
    return (predict_proba(X, coeffs) > thres).astype(int)


def cost(X, y, coeffs, lam=0):
    """
    Calculate the logistic cost function of the data with the given
    coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    y: A 1 dimensional numpy array.  The actual class values of the response.
        Must be encoded as 0's and 1's.  Also, must align properly with X and
        coeffs.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X, y, and coeffs must align.

    Returns
    -------
    logistic_cost: The computed logistic cost.
    """
    y_predicted = predict_proba(X,coeffs)
    costs = y*np.log(y_predicted) + (1-y)*np.log(1-y_predicted) + lam*(coeffs**2)
    return  -sum(costs)


def gradient(X, y, coeffs, lam=0):
    """
    Calculate the descent gradient of the data with the given
    coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    y: A 1 dimensional numpy array.  The actual class values of the response.
        Must be encoded as 0's and 1's.  Also, must align properly with X and
        coeffs.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X, y, and coeffs must align.

    Returns
    -------
    descent gradient.
    """
    y_predicted = predict_proba(X, coeffs)
    p_diffs = X.T.dot(y_predicted - y) + 2*lam*coeffs
    return p_diffs

def add_intercept(X):
    """Add an intercept column to a matrix X.

    Parameters
    ----------
    X: A two dimensional numpy array.

    Returns
    -------
    X: The original matrix X, but with a constant column of 1's appended.
    """
    return np.c_[np.ones(X.shape[0]), X]


def standardize(X):
    """Standardize each column of the input about mean and
    scaled by the standard deviation.

    Parameters
    ----------
    X: A numpy array

    Returns
    -------
    X: The standardized array X
    """
    col_mean = X.mean(axis = 0)
    col_std = X.std(axis=0)
    return (X - col_mean) / col_std


class LogisticRegressionFuncTest(unittest.TestCase):
    'Unit Tests for each relevant class'

    def test_hyp(self):
        X = -100
        soln = hyp(X)
        self.assertAlmostEqual(soln, 1.)
        X = 0
        soln = hyp(X)
        self.assertEqual(soln, 0.5)
        X = 100
        soln = hyp(X)
        self.assertAlmostEqual(soln, 0.)
        return

    def test_predict_proba(self):
        X = np.array([[0, 1], [2, 2], [3, 0]])

        coeffs = np.ones(2)
        soln = predict_proba(X, coeffs)
        check = np.array([0.731058578630005, 0.982013790037908, 0.952574126822433])
        self.assertAlmostEqual(soln[0], check[0])
        self.assertAlmostEqual(soln[1], check[1])
        self.assertAlmostEqual(soln[2], check[2])
        return

    def test_predict(self):
        X = np.array([[0], [10**3], [-1*10**3]])
        coeffs = np.ones((3,1))
        soln = predict(X, coeffs)
        check = np.array([0, 1, 0])
        self.assertAlmostEqual(soln[0], check[0])
        self.assertAlmostEqual(soln[1], check[1])
        self.assertAlmostEqual(soln[2], check[2])
        return

    def test_cost(self):
        X = np.array([[0, 1], [2, 2], [3, 0]])
        y = np.array([1,0,0])
        coeffs = np.ones(2)
        soln = cost(X, y, coeffs)
        check = 7.37999896700978
        self.assertAlmostEqual(soln, check)
        return

    def test_gradient(self):
        X = np.array([[0, 1], [2, 2], [3, 0]])
        y = np.array([1,0,0])
        coeffs = np.ones(2)
        soln = gradient(X, y, coeffs)
        check = np.array([4.82174996054312, 1.69508615870582])
        self.assertAlmostEqual(soln[0], check[0])
        self.assertAlmostEqual(soln[1], check[1])
        return

    def test_add_intercept(self):
        X = np.array([[5, 5, 5], [6, 6, 6]])
        soln = add_intercept(X)
        check = np.array([[1, 5, 5, 5], [1, 6, 6, 6]])
        self.assertTrue(np.array_equal(soln, check))
        return

    def test_standardize(self):
        X = np.array([5, 5, 6, 6])
        soln = standardize(X)
        check = np.array([-1, -1, 1, 1])
        self.assertTrue(np.array_equal(soln, check))
        return

if __name__ == '__main__':
    unittest.main()
