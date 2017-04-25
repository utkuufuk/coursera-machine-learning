function [J, gradient] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

    % calculate unregularized cost
    sampleSize = length(y); 
    predictions = X * theta;
    squaredErrors = (predictions - y) .^ 2;
    J = 1 / (2 * sampleSize) * sum(squaredErrors);

    % add regularization term
    regularization = lambda * sum(theta(2:end) .^ 2) / (2 * sampleSize);
    J = J + regularization;
    
    % calculate the regularized gradient
    gradient = ((X' * (predictions - y)) + (lambda * theta)) / sampleSize;
    gradient(1) = sum((predictions - y)) / sampleSize;
end
