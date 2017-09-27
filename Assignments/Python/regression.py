import numpy as np

def cost(X, y, theta):
    predictions = X @ theta
    return np.sum(np.square(predictions - y)) / (2 * len(y))

def gradient_descent(X, y, alpha, num_iters):
    num_features = X.shape[1]
    theta = np.zeros(num_features)
    cost_history = np.zeros(num_iters)
    for n in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = X.transpose() @ errors
        theta -= alpha * gradient / len(y)
        cost_history[n] = cost(X, y, theta)
    return theta, cost_history

