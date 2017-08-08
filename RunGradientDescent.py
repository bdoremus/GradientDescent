import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.GradientDescent as G
import src.logistic_regression_functions as f
from sklearn.linear_model import LogisticRegression


from sklearn.datasets import make_classification


def plot_data(X, y):
    """
    Given data with two features and a binary classification,
    plot them on the same graph with each classification made
    visually distinctive.
    """
    # split into two sets of datasets
    set1 = X[y==1]
    set0 = X[y==0]

    # plot first set with green circles
    plt.scatter(set1[:,0], set1[:,1], color='g', marker='o')

    # plot second set with red squares
    plt.scatter(set0[:,0], set0[:,1], color='r', marker='s')

    # Classifier based on SKLearn's coefficients
    beta = [0.18567507,  2.3793796,   0.36533597]
    x = np.linspace(-1, 1, 10)
    z = -1*((beta[0] + beta[1]*x)/beta[2])
    plt.plot(x, z, color='b', label="SKLearn Classifier boundary")
    plt.legend()
    plt.show()
    pass


def whole_batch_gradient_descent(X, y):
    """
    Given a training set of data with two features and a binary classification,
    use Gradient Descent to create an optimized mode for classification.
    Compare these results to SKLearn's coefficients.
    """

    gd = G.GradientDescent(f.cost, f.gradient, f.predict, alpha=0.01, num_iterations=10000)
    X = f.standardize(X.copy())
    X = f.add_intercept(X)
    gd.fit(X, y)
    # predictions = gd.predict(X)

    return gd.coeffs


def stochastic_gradient_descent(X, y):
    """
    Given a training set of data with two features and a binary classification,
    use Stochastic Gradient Descent to create an optimized mode for classification.
    Compare these results to SKLearn's coefficients.
    """
    sgd = G.GradientDescent(f.cost, f.gradient, f.predict, alpha=0.01, num_iterations=10000)
    X = f.standardize(X.copy())
    X = f.add_intercept(X)
    sgd.fit_stochastic(X, y)
    # predictions = sgd.predict(X)

    return sgd.coeffs



if __name__ == '__main__':
    # a flag to repress outputs
    show_all = True

    ##############
    ### Part 1 ###
    ##############
    # Make some random classification data
    # Eyeball a line of best fit. ( vertical line at x = -0.5)
    # Later, calculate it using GradientDescent
    X, y = make_classification(n_samples=100,
                                n_features=2,
                                n_informative=2,
                                n_redundant=0,
                                n_classes=2,
                                random_state=0)
    if show_all:
        plot_data(X, y)

    ##############
    ### Part 2 ###
    ##############
    # Get coefficients using Whole Batch Gradient Descent
    if show_all:
        print("Whole batch GD: ",whole_batch_gradient_descent(X, y))

    ##############
    ### Part 3 ###
    ##############
    # Get coefficients using Stochastic Gradient Descent
    if show_all:
        print("Stochastic Gradient Descent: ", stochastic_gradient_descent(X, y))

    # Check our model against SKLearn's
    model = LogisticRegression(fit_intercept=True)
    model.fit(X, y)
    if show_all:
        print("SKLearn: ",model.coef_)
