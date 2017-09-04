%% Machine Learning Online Class - Exercise 5 | Regularized Linear Regression & Bias-Variance
clear; close all; clc;
addpath(genpath('../common'))

%% =========== Part 1: Loading and Visualizing Data =============
load('data5.mat');
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

%% =========== Part 2: Unregularized Linear Regression =============
% train unregularized linear regression
theta = trainLinearRegression([ones(length(y), 1) X], y, 0);
predictions = [ones(length(y), 1) X] * theta;

% plot linear fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, predictions, '--', 'LineWidth', 2)
hold off;

% Since the model is underfitting the data, we expect to see a graph with "high bias"
learningCurve([ones(length(y), 1) X], y, [ones(length(yval), 1) Xval], yval, 0);


%% =========== Part 3: Feature Mapping for Polynomial Regression =============
% Map X onto Polynomial Features and Normalize
p = 8;
X_poly = polyFeatures(X, p);
[X_poly, means, stDevs] = featureNormalize(X_poly);
X_poly = [ones(length(y), 1), X_poly];

% Map X_poly_test and normalize (using means and stDevs)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, means);
X_poly_test = bsxfun(@rdivide, X_poly_test, stDevs);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];

% Map X_poly_val and normalize (using means and stDevs)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, means);
X_poly_val = bsxfun(@rdivide, X_poly_val, stDevs);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];

%% =========== Part 4: Learning Curve for Polynomial Regression ===========
% train polynomial regression with lambda = 1
lambda = 1;
[theta] = trainLinearRegression(X_poly, y, lambda);

% plot training data and fit
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), means, stDevs, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

% plot the learning curve
learningCurve(X_poly, y, X_poly_val, yval, lambda);

%% ============ Part 5: Validation Curve for Selecting Lambda =============
% select several lambda values to try
lambdaValues = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
trainingError = zeros(length(lambdaValues), 1);
validationError = zeros(length(lambdaValues), 1);

% compute training and validation errors for each lambda
for i = 1:length(lambdaValues)
    % compute errors without regularization
    theta = trainLinearRegression(X_poly, y, lambdaValues(i));
    trainingError(i) = regressionCost(X_poly, y, theta, 0);
    validationError(i) = regressionCost(X_poly_val, yval, theta, 0);
end

% choose lambda with the lowest validation error
[~, minIndex] = min(validationError);
bestLambda = lambdaValues(minIndex);

% plot the validation curve
plot(lambdaValues, trainingError, lambdaValues, validationError);
legend('Training', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

% compute test error using the selected lambda
theta = trainLinearRegression(X_poly, y, bestLambda);
testError = regressionCost(X_poly_test, ytest, theta, 0); % no regularization
fprintf('test error = %f (at lambda = %f)\n', testError, bestLambda);
plot(Xtest, ytest, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(Xtest), max(Xtest), means, stDevs, theta, p);
