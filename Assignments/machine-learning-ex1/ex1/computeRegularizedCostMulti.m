function J = computeRegularizedCostMulti(X, y, theta, lambda)

    m = length(y);  % number of training examples
    predictions = X * theta;
    squaredErrors = (predictions - y) .^ 2;
    theta(1) = [];
    thetaSquared = lambda * (theta .^ 2);
    J = (sum(squaredErrors) + sum(thetaSquared)) / (2 * m);
end
