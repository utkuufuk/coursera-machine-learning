function [lambdaValues, trainError, validationError] = validationCurve(X, y, valX, valY)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

    % Selected values of lambda
    lambdaValues = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
    trainError = zeros(length(lambdaValues), 1);
    validationError = zeros(length(lambdaValues), 1);

    for i = 1:length(lambdaValues)
        
        theta = trainLinearReg(X, y, lambdaValues(i));
        trainError(i) = regressionCost(X, y, theta, 0);
        validationError(i) = regressionCost(valX, valY, theta, 0);
    end
end
