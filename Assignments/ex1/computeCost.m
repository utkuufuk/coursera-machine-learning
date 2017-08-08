function J = computeCost(X, y, theta)
% Compute cost for linear regression 

    numExamples = length(y);
    predictions = X * theta;
    squaredErrors = (predictions - y) .^ 2;
    J = sum(squaredErrors) / (2 * numExamples); 
end
