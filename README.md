# GradientDescent
The following program was an exercise in writing a Gradient Descent and Stochastic Gradient Descent algorithm.

# Goals
Understand the nuances of popular Gradient Descent algorithms, the advantages and disadvantages of each algorithm, and when it is appropriate to use each model.

# Process
1. Outside of this code, create a simple test case and solved by hand.  
Use those results to create unit tests for logistic_regression_functions.py.

1. Generate a random classification problem with two features.  
Plot the values to estimate the coefficients of a solution.

1. Write a Gradient Descent (GD) class to run a Whole Batch GD.  
Compare the results to SKLearn as a gold standard.

1. Add to the Gradient Descent (GD) class to run a Stochastic GD.  
Compare the results to SKLearn as a gold standard.

# Conclusions
All of my models were similar, but inferior to SKLearn's in both accuracy and efficiency. I assume there were significant optimization under the hood of SKLearn, likely in using Adagrad to adjust the learning rate between each iteration.

Whole Batch GD is very effective and computationally reasonable for smaller data sets.  Its limitations are twofold:
  1. Because it is run on the entire data set, it is unreasonble to use this method with a large number of tests or many features (>10,000)
  1. Because it must be re-run on the entire data set every time, it is only appropriate to use this method when new data will not be reincorporated into the model.

Stochastic Gradient Descent is computationally expensive given that everything is recalculated for each individual test.  Despite that severe drawback, it does have its uses:
  1. It is most appropriate with smaller data sets, due to the large number of computations needed for each iteration
  1. It is very useful when new data needs to be encorporated into the model
  1. Very noisy when converging

As a middle ground between the two, Small Batch Gradient Descent allows for relatively inexpensive future updates to the model, combined with th computational advantage gained by processing larger groups of data in each iteration.
'''
