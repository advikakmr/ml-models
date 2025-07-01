import numpy as np
import matplotlib.pyplot as plt

from utils import *

# m = 5
X_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([4, 7, 10, 13, 16]) 
# n = 5
X_test = np.array([2.3, 4.4, 5, 6, 6.5])
y_test = np.array([10.9, 17.2, 19, 22, 23.5])

model = NeuralNetwork(X_train, y_train, X_test, y_test)

prediction = model.predict_avg(100)
print(f"prediction: {prediction}\nactual: {y_test}\navg error: {model.error_calc(prediction)}\n")

# graphing
plt.plot(X_train, y_train, "rs", X_test, y_test, "bs", X_test, prediction, "g--")
plt.xlabel("X")
plt.ylabel("y")
plt.show()