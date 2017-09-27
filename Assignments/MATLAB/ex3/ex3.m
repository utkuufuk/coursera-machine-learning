%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
clear; close all; clc;
addpath(genpath('../common'))
%% =========== Part 1: Loading and Visualizing Data =============
load('data3.mat');
[numExamples, numFeatures] = size(X);
numClasses = 10;

% Randomly select 100 data points to display
randIndices = randperm(numExamples);
displayDigits(X(randIndices(1:100), :));
%% ============ Part 2: One-vs-All Training ============
lambda = 0.1;

% initialize a classifier for each class (1 to 10)
classifiers = zeros(numClasses, numFeatures + 1);

% Add intercept term to X
X = [ones(length(y), 1) X];

% train each classifier
for i = 1:numClasses
    
    class = y == i;
    theta = zeros(numFeatures + 1, 1);

    options = optimset('GradObj', 'on', 'MaxIter', 50);
    theta = fmincg(@(t)(classificationCost(t, X, class, lambda)), theta, options);
    classifiers(i, 1:numFeatures + 1) = theta;
end
%% ================ Part 3: Predict for One-Vs-All ================
probabilities = sigmoid(X * classifiers');      % probability of each class
[~, predictions] = max(probabilities, [], 2);   % most probable class for each example
fprintf('\nTraining Set Accuracy: %f\n', mean(predictions == y) * 100);
