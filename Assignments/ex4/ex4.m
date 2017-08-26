%% Machine Learning Online Class - Exercise 4 Neural Network Learning
clear; close all; clc;
addpath(genpath('../common'))

%% Part 1: Loading and Visualizing Data
load('data4.mat');
[numExamples, numInputUnits] = size(X);
numHiddenUnits = 25;   % number of neurons inside the hidden layer
numClasses = 10;    % 10 labels, from 1 to 10 (we mapped "0" to "10")

% Randomly select 100 data points to display
sel = randperm(numExamples);
sel = sel(1:100);
displayDigits(X(sel, :));

%% Part 2: Training the Network
% initialize parameters
hiddenTheta = randInitLayer(numInputUnits, numHiddenUnits);
outputTheta = randInitLayer(numHiddenUnits, numClasses);
unrolledTheta = [hiddenTheta(:); outputTheta(:)];

options = optimset('MaxIter', 400); % increase MaxIter to see how more training helps
lambda = 1;                         % also try different values of lambda

% create "short hand" for the cost function to be minimized
costFunction = @(p) nnCost(p, numHiddenUnits, numClasses, X, y, lambda);

% now, costFunction is a function that takes in only one argument (the NN parameters)
[unrolledTheta, cost] = fmincg(costFunction, unrolledTheta, options);

% Obtain Theta1 and Theta2 back from unrolledTheta
[Theta1, Theta2] = recoverTheta(unrolledTheta, numHiddenUnits, numInputUnits, numClasses);
             
%% Part 3: Prediction
predictions = nnPredict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(predictions == y) * 100);

% randomly permute examples
rp = randperm(length(y));

% run through the examples one at the a time to see what it is predicting.
for i = 1:length(y)

    fprintf('\nDisplaying Example Image\n');
    displayDigits(X(rp(i), :));

    predictions = nnPredict(Theta1, Theta2, X(rp(i), :));
    fprintf('\nNN Prediction: %d (digit %d)\n', predictions, mod(predictions, 10));
    
    s = input('Paused - press enter to continue, q to exit:', 's');
    
    if s == 'q'
      break
    end
end
