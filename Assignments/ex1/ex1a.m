%% Machine Learning Online Class - Exercise 1: Linear Regression
clear; close all; clc;
addpath(genpath('../common'))
%% =================== Part 1: Gradient descent ===================
% Read and plot the data
data = load('data1a.txt');
X = data(:, 1);             % population size in 10,000s
y = data(:, 2);             % profit in $10,000s
numExamples = length(y);    % number of training examples

% Plot data
figure;
plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('City population in 10,000s');

% Initialize model parameters
X = [ones(numExamples, 1), X];  % add a column of ones to x
theta = zeros(2, 1);            % initialize model parameters
fprintf('The initial cost is: %f\n', regressionCost(X, y, theta, 0));    

% Gradient descent settings
numIters = 600;
alpha = 0.02;

% Run gradient descent
[theta, costHistory] = gradientDescent(X, y, theta, alpha, 0, numIters);
fprintf('The final cost is: %f\n', costHistory(end));
fprintf('Theta found by gradient descent: %f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:, 2), X * theta, '-');
legend('Training data', 'Linear regression');
hold off; % don't overlay any more plots on this figure

% Plot the convergence graph
figure;
plot(1:numel(costHistory), costHistory, 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
title(strcat('learning rate = ', {' '}, num2str(alpha)));

% Predict profit for population size 70,000
prediction = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', prediction * 10000);

%% ============= Part 2: Visualizing J(theta_0, theta_1) =============
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = regressionCost(X, y, t, 0);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
