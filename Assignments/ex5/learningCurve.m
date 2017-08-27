function learningCurve(X, y, valX, valY, lambda)

    trainError = zeros(length(y), 1);
    validationError = zeros(length(y), 1);
    
    for i = 1:length(y)
        theta = trainLinearRegression(X(1:i, :), y(1:i), lambda);
        
        % we must compute the errors without regularization
        trainError(i) = regressionCost(X(1:i, :), y(1:i), theta, 0);
        validationError(i) = regressionCost(valX, valY, theta, 0);
    end
    
    plot(1:length(y), trainError, 1:length(y), validationError);
    legend('Training', 'Cross Validation')
    xlabel('Number of training examples')
    ylabel('Error')
end
