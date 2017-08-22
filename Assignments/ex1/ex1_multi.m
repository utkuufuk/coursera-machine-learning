%% Machine Learning Online Class - Exercise 1: Linear regression with multiple variables

%% ================ Part 1: Feature Normalization ================

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.2;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, 0, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
normalizedFeature1 = (1650 - mu(1)) / sigma(1);
normalizedFeature2 = (3 - mu(2)) / sigma(2);
price = [1, normalizedFeature1, normalizedFeature2] * theta;
fprintf('Predicted price of a 1650 sq-ft, 3 br house:\n $%f\n', price);

%% ================ Part 3: Normal Equations ================

data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = [1, 1650, 3] * theta;
fprintf('Predicted price of a 1650 sq-ft, 3 br house:\n $%f\n', price);

