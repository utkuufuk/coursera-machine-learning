import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import regression as reg

# training set
train_set = np.loadtxt('water_train_set.txt', delimiter=",")
x_train = train_set[:, 0]
y_train = train_set[:, 1]

# validation set
val_set = np.loadtxt('water_val_set.txt', delimiter=",")
x_val = val_set[:, 0]
y_val = val_set[:, 1]

# test set
test_set = np.loadtxt('water_test_set.txt', delimiter=",")
x_test = test_set[:, 0]
y_test = test_set[:, 1]

# plot training data
plt.scatter(x_train, y_train, marker="x", s=40, c='red')
plt.xlabel("Change in water level", fontsize=14)
plt.ylabel("Water flowing out of the dam ", fontsize=14)

X_train = np.ones(shape=(len(x_train), 2))
X_train[:, 1] = x_train

def train_linear_regression(X, y):
    theta = np.zeros(X.shape[1])
    return opt.fmin_cg(reg.cost, theta, reg.cost_gradient, (X, y))

theta = train_linear_regression(X_train, y_train)
predictions = X_train @ theta
plt.plot(X_train[:, 1], predictions, linewidth=2)
plt.show()






