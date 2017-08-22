function [theta, costHistory] = gradientDescent(X, y, theta, alpha, lambda, numIters)

    costHistory = zeros(numIters, 1);
    
    
    for iterationIndex = 1:numIters
        
        % calculate predictions with the parameters calculated in the previous iteration
        predictions = X * theta;  
        
        % add regularization term
        regularization = lambda * theta;
        regularization(1) = 0;  % don't penalize the intercept term.
        
        % update theta
        descent = transpose(X) * (predictions - y) + regularization;
        theta = theta - (alpha * descent / length(y));
        
        % record cost
        costHistory(iterationIndex) = computeCost(X, y, theta, lambda);
    end
end
