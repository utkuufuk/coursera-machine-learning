function [J, grad] = classificationCost(theta, X, y, lambda)

    sampleSize = length(y); 
    predictions = sigmoid(X * theta);
    errorPositives = - y .* log(predictions);
    errorNegatives = (y - 1) .* log(1 - predictions);
    regularization = lambda * sum(theta(2:end) .^ 2) / 2;
    J = (regularization + sum(errorPositives) + sum(errorNegatives)) / sampleSize;
    
    theta(1) = 0; % don't regularize the intercept term
    grad = ((X' * (predictions - y)) + (lambda * theta)) / sampleSize;
end

