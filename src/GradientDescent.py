import numpy as np
import unittest


class GradientDescent(object):
    """Perform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict_func,
                 alpha=0.01,
                 num_iterations=10000):
        """Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimizaiton has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.alpha = alpha
        self.num_iterations = num_iterations


    def fit(self, X, y, step_size=0.0001):
        """Run the gradient descent algorithm for num_iterations repititions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.
        step_size: a float.  The learning rate.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        self.coeffs = np.zeros(X.shape[1])
        for k in range(self.num_iterations):
            new_coeff = self.coeffs - self.alpha * self.gradient(X, y, self.coeffs, lam=1)

            # Stop early if all coefficients have converged
            if np.all(new_coeff-self.coeffs < step_size):
                break
            self.coeffs = new_coeff
        return


    def fit_stochastic(self, X, y, step_size=0.001):
        """Run the gradient descent algorithm for num_iterations repititions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.
        step_size: a float.  The learning rate.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        self.coeffs = np.ones(X.shape[1])

        # shuffle the indexes
        i_list = np.random.permutation(X.shape[0])

        overage = 0
        for k in range(self.num_iterations):
            # save the old coefficients to check for convergence later
            old_coeffs = self.coeffs

            for index in i_list:
                # update the coefficients
                self.coeffs = self.coeffs - self.alpha*self.gradient(X[index, :], y[index], self.coeffs, lam=1)

            # Stop early if all coefficients have converged
            if np.all(old_coeffs-self.coeffs < step_size):
                break
        return




    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        return self.predict_func(X, self.coeffs)
