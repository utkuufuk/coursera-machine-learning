function cost = computeCost(X, y, theta, lambda) 
    predictions = X * theta;
    squaredErrors = (predictions - y) .^ 2;
    theta(1) = [];
    thetaSquared = lambda * (theta .^ 2);
    cost = (sum(squaredErrors) + sum(thetaSquared)) / (2 * length(y));
end