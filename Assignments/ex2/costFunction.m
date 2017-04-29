function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

    sampleSize = length(y); 
    predictions = sigmoid(X * theta);
    errorPositives = - y .* log(predictions);
    errorNegatives = (y - 1) .* log(1 - predictions);
	
    J = (sum(errorPositives) + sum(errorNegatives)) / sampleSize;
	grad = (X' * (predictions - y)) / sampleSize;
end
