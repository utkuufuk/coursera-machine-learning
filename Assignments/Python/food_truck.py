import numpy as np
import matplotlib.pyplot as plt

import regression as reg

# read & plot training data
data = np.loadtxt('food_truck.txt', delimiter=",")
x = data[:, 0]
y = data[:, 1]

plt.scatter(x, y, c='red')
plt.title("Food Truck Data Set", fontsize=20)
plt.xlabel("City Population in 10,000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10,000s", fontsize=14)
plt.axis([4, 24, -5, 25])

# add a column of ones to x and obtain feature matrix X
X = np.ones(shape=(len(x), 2))
X[:, 1] = x

# initialize model parameters
num_features = X.shape[1]
theta = np.zeros(num_features)

# compute the initial cost
print('The initial cost is:', reg.cost(X, y, theta))

# train model and plot fit
theta, _ = reg.gradient_descent(X, y, 0.02, 600)
predictions = X @ theta
plt.plot(X[:, 1], predictions, linewidth=2)

# train models using different learning rates and plot convergence  
plt.figure()
_, cost_history = reg.gradient_descent(X, y, 0.01, 1200)
plt.plot(cost_history, linewidth=2)

_, cost_history = reg.gradient_descent(X, y, 0.012, 1200)
plt.plot(cost_history, linewidth=2)

_, cost_history = reg.gradient_descent(X, y, 0.014, 1200)
plt.plot(cost_history, linewidth=2)

_, cost_history = reg.gradient_descent(X, y, 0.016, 1200)
plt.plot(cost_history, linewidth=2)

_, cost_history = reg.gradient_descent(X, y, 0.018, 1200)
plt.plot(cost_history, linewidth=2)

_, cost_history = reg.gradient_descent(X, y, 0.02, 1200)
plt.plot(cost_history, linewidth=2)

plt.title("Convergence plots for different learning rates")
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.axis([0, 1200, 4, 6])
plt.grid()
plt.legend(["0.010", "0.012", "0.014", "0.016", "0.018", "0.020"])

# train a model using a large learning rate and plot convergence
_, cost_history = reg.gradient_descent(X, y, 0.025, 50)
plt.figure()
plt.plot(cost_history, linewidth=2)
plt.title("Convergence plot for a learning rate of 0.025")
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.axis([0, 50, 0, 6000])
plt.grid()

# make a prediction based on the optimized model
theta, _ = reg.gradient_descent(X, y, 0.02, 600)
test_example = np.array([1, 7])
prediction = test_example @ theta
print('For population = 70,000, we predict a profit of $' + str(prediction * 10000));

# display plots
plt.show()
