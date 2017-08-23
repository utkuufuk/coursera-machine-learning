%% Machine Learning Online Class - Exercise 1: Linear regression with multiple variables

%% ================ Part 1: Feature Normalization ================
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
numExamples = length(y);

% Scale features and set them to zero mean
[X, means, stDevs] = featureNormalize(X);

% Add intercept term to X
X = [ones(numExamples, 1), X];

% Choose some alpha value
alpha = 0.2;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, costHistory] = gradientDescent(X, y, theta, alpha, 0, num_iters);
fprintf('The final cost is: %f\n', costHistory(end));

% Plot the convergence graph
figure;
plot(1:numel(costHistory), costHistory, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
normalizedFeature1 = (1650 - means(1)) / stDevs(1);
normalizedFeature2 = (3 - means(2)) / stDevs(2);
price = [1, normalizedFeature1, normalizedFeature2] * theta;
fprintf('Predicted price of a 1650 sq-ft, 3 br house:\n $%f\n', price);

%% ================ Part 3: Normal Equations ================
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
numExamples = length(y);

% Add intercept term to X
X = [ones(numExamples, 1) X];

% Calculate the parameters from the normal equation
theta = normalEquation(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = [1, 1650, 3] * theta;
fprintf('Predicted price of a 1650 sq-ft, 3 br house:\n $%f\n', price);

