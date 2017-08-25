function [cost, gradient] = regressionCost(X, y, theta, lambda)

    predictions = X * theta;
    squaredErrors = (predictions - y) .^ 2;
    thetaSquared = lambda * (theta(2:end) .^ 2);
    cost = (sum(squaredErrors) + sum(thetaSquared)) / (2 * length(y));
    
    theta(1) = 0; % don't regularize the intercept term
    gradient = (X' * (predictions - y) + lambda * theta) / length(y);
end
