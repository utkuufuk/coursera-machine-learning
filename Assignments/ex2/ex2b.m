%% Machine Learning Online Class - Exercise 2b: Regularized Logistic Regression
data = load('data2b.txt');
X = data(:, 1:2); 
y = data(:, 3);

plotData(X, y);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;

% mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
theta = zeros(size(X, 2), 1);

% Set regularization parameter (you should experiment with this)
lambda = 1;

% Optimize
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J] = fminunc(@(t)(classificationCost(t, X, y, lambda)), theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(p == y) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
