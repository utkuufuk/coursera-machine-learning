function J = computeCostMulti(X, y, theta)
%Computes cost for multi-variate linear regression

    numExamples = length(y);
    predictions = X * theta;
    squaredErrors = (predictions - y) .^ 2;
    J = sum(squaredErrors) / (2 * numExamples);
end
