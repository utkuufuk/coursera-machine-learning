function [theta, costHistory] = gradientDescent(X, y, theta, alpha, lambda, numIters)

    costHistory = zeros(numIters, 1);
    
    for iterationIndex = 1:numIters
        
        predictions = X * theta;  
        regularization = lambda * theta;
        regularization(1) = 0;  % don't penalize the intercept term.
        descent = transpose(X) * (predictions - y) + regularization;
        theta = theta - (alpha * descent / length(y));
        costHistory(iterationIndex) = computeCost(X, y, theta, lambda);
    end
end
