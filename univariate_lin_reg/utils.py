import numpy as np
from numpy.typing import NDArray
import random

class NeuralNetwork:
    def __init__(self, X_train: NDArray[np.float64], y_train: NDArray[np.float64], X_test: NDArray[np.float64], y_test: NDArray[np.float64]):
        self.w: float = random.uniform(1.0, 100.0)
        self.b: float = random.uniform(1.0, 100.0)
        
        # shape: (m, ) - 1D vector of m training examples
        self.X_train = X_train 
        self.y_train = y_train 
        # shape: (n, ) - 1D vector of n test cases
        self.X_test = X_test
        self.y_test = y_test 

    # def _mean_normalization(self):
    #     mean = np.mean(self.X_train, axis=0)
    #     range = np.ptp(self.X_train, axis=0)
    #     for i in range(len(self.X_train)):
    #         self.X_train[i] = (self.X_train[i] - mean)/range

    # squared error cost function for linear regression
    def _lin_reg_cost(self) -> float:
        m = len(self.X_train) # number of training examples
        total_cost = 0
        for i in range(m):
            f_wb = (self.w * self.X_train[i]) + self.b
            loss = np.square(f_wb - self.y_train[i])
            total_cost += loss
        return total_cost/(2*m)

    def _partial_deriv(self, axis: int) -> float:
        m = len(self.X_train)
        deriv = 0
        for i in range(m):
            f_wb = (self.w * self.X_train[i]) + self.b
            if axis == 0: # w
                deriv += (f_wb - self.y_train[i]) * self.X_train[i]
            else: # b
                deriv += (f_wb - self.y_train[i])
        return deriv

    def _gradient_descent(self, alpha: float = 0.01, epsilon: float = 0.01):
        m = len(self.X_train)
        done = False
        while not done:
            cost = self._lin_reg_cost()
            w_partial_deriv = self._partial_deriv(axis=0)
            b_partial_deriv = self._partial_deriv(axis=1)
            self.w -= (alpha * w_partial_deriv)
            self.b -= (alpha * b_partial_deriv)
            new_cost = self._lin_reg_cost()
            if abs(new_cost - cost) < epsilon:
                done = True
    
    # def _sigmoid(self, layer: NDArray[np.float64]) -> NDArray[np.float64]:
    #     return 1/(1 + np.exp(layer))
    
    def predict(self) -> NDArray[np.float64]:
        self._gradient_descent()
        # print(f"w: {self.w}\nb:{self.b}\n")
        prediction = (self.w * self.X_test) + self.b
        return prediction
    
    def predict_avg(self, n) -> NDArray[np.float64]:
        res = []
        for _ in range(n):
            res.append(self.predict())
        print(f"[debug] Finished {n} iterations\n")
        return np.mean(res, axis=0)
    
    def error_calc(self, prediction: NDArray[np.float64]) -> float:
        error = []
        for i in range(len(prediction)):
            error.append(abs(prediction[i]-self.y_test[i]))
        return np.mean(error)