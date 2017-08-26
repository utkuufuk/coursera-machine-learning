%% Machine Learning Online Class - Exercise 4 Neural Network Learning
clear; close all; clc;
addpath(genpath('../../common'))
addpath(genpath('..'))

%% Part 1: Loading and Visualizing Data
load('data4.mat');
[numExamples, ~] = size(X);
hiddenUnits = 25;   % number of neurons inside the hidden layer
numClasses = 10;    % 10 labels, from 1 to 10 (we mapped "0" to "10")

%% Part 2: Testing
load('data4test.mat');
unrolledTheta = [Theta1(:); Theta2(:)];
cost = nnCost(unrolledTheta, hiddenUnits, numClasses, X, y, 0);
fprintf('Cost with lambda = 0: %f\n(this value should be about 0.287629)\n', cost);
cost = nnCost(unrolledTheta, hiddenUnits, numClasses, X, y, 1);
fprintf('Cost with lambda = 1: %f\n(this value should be about 0.383770)\n', cost);
cost  = nnCost(unrolledTheta, hiddenUnits, numClasses, X, y, 3);
fprintf('Cost with lambda = 3: %f\n(this value should be about 0.576051)\n\n', cost);
checkNNGradients;
checkNNGradients(3);
