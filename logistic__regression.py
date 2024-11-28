import numpy as np
import random

class LRClassifier:
    
    learning_rate = 0.01

    def __init__(self, _learning_rate=0.01):
        self.learning_rate = _learning_rate

    def sigmoid(self, x):
      return 1/(1+ np.exp(-1*x))

    def predict(self, x, beta):
        return self.sigmoid(np.dot(x,beta))

#Likelihood = sig(x.beta)^y * sig(x.beta)^(1-y)

    def _negative_log_L(self, x, y, beta):
        if y == 1:
                return -np.log(self.sigmoid(np.dot(x,beta)))
        else:
                return -1*np.log(1 - self.sigmoid(np.dot(x,beta)))
      
    def negative_log_L(self, xs, ys, beta):
        return sum( [ self._negative_log_L(x,y,beta=beta) for (x,y) in zip(xs, ys) ] )

    def _negative_log_partial(self, x ,y ,beta , j):
        return -(y - self.sigmoid(np.dot(x,beta)))*x[j]

    def _negative_log_gradient(self, x,y,beta) -> np.array:
        arr = [ self._negative_log_partial(x,y,beta,j) for j in range(len(beta)) ]
        return np.array(arr)

    def negative_log_gradient(self, xs, ys, beta : np.array):
        return sum( [ self._negative_log_gradient(x,y,beta) for (x,y) in zip(xs,ys) ] )
    
    def predict_beta(self, xs, ys):
        beta = [ random.random() for _ in range(30) ]

        for epoch in range(5000):

            gradient = self.negative_log_gradient(xs, ys, beta)
            beta = beta + (-1* self.learning_rate * gradient)
        
        return beta