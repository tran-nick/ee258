# perceptron.py
import numpy as np

class Perceptron(object):
   def __init__(self, rate = 0.01, niter = 10): # default class arguments
      self.rate = rate  # learning rate
      self.niter = niter # number of epochs

   def fit(self, X, y):
      """Fit training data
      X : Training vectors, X.shape : [#samples, #features]
      y : Target values, y.shape : [#samples]
      """

      # weights
      self.weight = np.zeros(1 + X.shape[1]) # weights and biases are initialized to 0 

      # Number of misclassifications
      self.errors = []  # Number of misclassifications (errors)

      for i in range(self.niter): # loop for epochs
         err = 0                  # reset error to 0 each iteration
         for xi, target in zip(X, y):
            # below three lines updates the weights based on the error (target - predicted), xi is the ith training data
            delta_w = self.rate * (target - self.predict(xi))
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w  # bias update
            
            err += int(delta_w != 0.0) # accumulate errors
         self.errors.append(err)  # save the error for ith epoch
      return self

   def net_input(self, X): # returns the v value 
      """Calculate net input"""
      return np.dot(X, self.weight[1:]) + self.weight[0]

   def predict(self, X): # returns the value after activation function
      """Return class label after unit step"""
      return np.where(self.net_input(X) >= 0.0, 1, -1)