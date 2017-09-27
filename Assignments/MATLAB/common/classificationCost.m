function [cost, gradient] = classificationCost(theta, X, y, lambda)

    predictions = sigmoid(X * theta);
    error = -y .* log(predictions) - (1 - y) .* log(1 - predictions);
    theta(1) = 0; % don't regularize the intercept term
    regularization = lambda * sum(theta .^ 2) / 2;
    cost = (regularization + sum(error)) / length(y);
    gradient = ((X' * (predictions - y)) + (lambda * theta)) / length(y);
end

