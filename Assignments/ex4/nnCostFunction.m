function [J, gradient] = nnCostFunction(unrolledTheta, numberOfFeatures, hiddenLayerSize, numberOfClasses, X, y, lambda)
% Implements the neural network cost function for a two layer neural network which performs classification
% [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, numberOfClasses, X, y, lambda) computes the cost and gradient of the neural network. 
% The parameters for the neural network are "unrolled" into the vector unrolledTheta and need to be converted back into the weight matrices. 
% The returned gradient parameter should be an "unrolled" vector of the partial derivatives of the neural network.

    %% Reshape unrolledTheta back into the parameters hiddenTheta and outputTheta, the weight matrices for our 2-layer neural network
    
    % hiddenTheta is a hiddenLayerSize by (numberOfFeatures +1) matrix 
    hiddenThetaRows = hiddenLayerSize;
    hiddenThetaCols = (numberOfFeatures + 1);
    hiddenThetaSegment = unrolledTheta(1:hiddenThetaRows * hiddenThetaCols);
    hiddenTheta = reshape(hiddenThetaSegment, hiddenThetaRows, hiddenThetaCols);
    
    % outputTheta is a numberOfClasses by (hiddenLayerSize + 1) matrix
    outputThetaRows = numberOfClasses;
    outputThetaCols = (hiddenLayerSize + 1);
    outputThetaSegment = unrolledTheta((1 + (hiddenThetaRows * hiddenThetaCols)):end);
    outputTheta = reshape(outputThetaSegment, outputThetaRows, outputThetaCols);
    
    %% Recode the output labels into a vector with dimensions equal to the output layer size
    
    sampleSize = length(y);
    I = eye(numberOfClasses);
    Y = zeros(sampleSize, numberOfClasses);
    
    for i = 1:sampleSize
        
        Y(i, :) = I(y(i), :);
    end

    %% Calculate the cost J
    
    % feedforward the neural network
    inputLayer = [ones(sampleSize, 1) X];
    z2 = inputLayer * hiddenTheta';
    hiddenLayer = [ones(size(z2, 1), 1) sigmoid(z2)];
    z3 = hiddenLayer * outputTheta';
    outputLayer = sigmoid(z3);
    predictions = outputLayer;

    % calculate the unregularized cost
    errors = (-Y) .* log(predictions) - (1 - Y) .* log(1 - predictions);
    J = sum(sum(errors, 2)) / sampleSize;
    
    % add the regularization penalty
    regularization = sum(sum(hiddenTheta(:, 2:end) .^ 2, 2)) + sum(sum(outputTheta(:, 2:end) .^ 2, 2));
    J = J + lambda * regularization / (2 * sampleSize);
    
    %% Backpropagation
    
    % calculate sigmas
    sigma3 = outputLayer - Y;
    sigma2 = (sigma3 * outputTheta) .* sigmoidGradient([ones(size(z2, 1), 1) z2]);
    sigma2 = sigma2(:, 2:end);

    % accumulate gradients
    delta1 = (sigma2' * inputLayer);
    delta2 = (sigma3' * hiddenLayer);

    % calculate regularized gradient
    p1 = (lambda / sampleSize) * [zeros(size(hiddenTheta, 1), 1) hiddenTheta(:, 2:end)];
    p2 = (lambda / sampleSize) * [zeros(size(outputTheta, 1), 1) outputTheta(:, 2:end)];
    hiddenThetaGradient = delta1 ./ sampleSize + p1;
    outputThetaGradient = delta2 ./ sampleSize + p2;
    
    % Unroll gradients
    gradient = [hiddenThetaGradient(:); outputThetaGradient(:)];
end
