function [theta, costHistory] = gradientDescent(X, y, theta, alpha, numIters)

    costHistory = zeros(numIters, 1);

    for iterationIndex = 1:numIters
        
        % calculate predictions with the parameters calculated in the previous iteration
        predictions = X * theta;  
        
        regularization = lambda * theta;
        regularization(1) = 0;  % don't penalize the intercept term.
        
        descent = transpose(X) * (predictions - y) + regularization;
        theta = theta - (alpha * descent / length(y));
        costHistory(iterationIndex) = computeCost(X, y, theta);     % Save the cost J in every iteration
    end
end
