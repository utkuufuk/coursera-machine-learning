function J = computeCost(X, y, theta)
    predictions = X * theta;                    % compute predictions for each training example
    squaredErrors = (predictions - y) .^ 2;     % use the difference between our predictions to compute the squared errors
    J = sum(squaredErrors) / (2 * length(y));   % compute total cost by averaging the squared errors
end
