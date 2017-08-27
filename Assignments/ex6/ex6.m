%% Machine Learning Online Class - Exercise 6 | Support Vector Machines
clear; close all; clc;
addpath(genpath('../common'))

%% =============== Part 1: Loading and Visualizing Data ================
load('data6a.mat');
plotData(X, y);

%% ==================== Part 2: Training Linear SVM ====================
% try to change the C value and see how the decision boundary varies
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

%% =============== Part 3: Visualizing Dataset 2 ================
load('data6b.mat');
plotData(X, y);

%% ========== Part 4: Training SVM with RBF Kernel (Dataset 2) ==========
% SVM Parameters
C = 1; 
sigma = 0.1;
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

%% =============== Part 5: Visualizing Dataset 3 ================
load('data6c.mat');
plotData(X, y);

%% ========== Part 6: Training SVM with RBF Kernel (Dataset 3) ==========
% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
