function theta = trainLinearRegression(X, y, lambda)
% Trains linear regression using fmincg

    theta = zeros(size(X, 2), 1);
    options = optimset('MaxIter', 200, 'GradObj', 'on');
    costFunction = @(t) regressionCost(X, y, t, lambda);
    theta = fmincg(costFunction, theta, options);
end
