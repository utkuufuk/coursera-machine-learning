function [J, grad] = costFunctionReg(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    % calculate unregularized cost
    sampleSize = length(y); 
    predictions = sigmoid(X * theta);
    errorPositives = - y .* log(predictions);
    errorNegatives = (y - 1) .* log(1 - predictions);
    J = (sum(errorPositives) + sum(errorNegatives)) / sampleSize;
    
    % add regularization term
    regularization = lambda * sum(theta(2:end) .^ 2) / (2 * sampleSize);
    J = J + regularization;
    
    grad = ((X' * (predictions - y)) + (lambda * theta)) / sampleSize;
    grad(1) = sum((predictions - y)) / sampleSize;  % don't regularize theta(0)
end

