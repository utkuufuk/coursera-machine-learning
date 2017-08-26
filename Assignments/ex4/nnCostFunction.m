function [cost, gradient] = nnCostFunction(theta, hiddenLayerSize, numClasses, X, y, lambda)
% Implements the neural network cost function for a two layer neural network which performs classification
% [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, numberOfClasses, X, y, lambda) computes the cost and gradient of the neural network. 
% The parameters for the neural network are "unrolled" into the vector unrolledTheta and need to be converted back into the weight matrices. 
% The returned gradient parameter should be an "unrolled" vector of the partial derivatives of the neural network.
    
    %% Reshape unrolledTheta back into the parameters hiddenTheta and outputTheta
    [numExamples, numFeatures] = size(X);
    
    % hiddenTheta is a hiddenLayerSize by (numberOfFeatures + 1) matrix 
    hiddenThetaSegment = theta(1:hiddenLayerSize * (numFeatures + 1));
    hiddenTheta = reshape(hiddenThetaSegment, hiddenLayerSize, (numFeatures + 1));
    
    % outputTheta is a numberOfClasses by (hiddenLayerSize + 1) matrix
    outputThetaSegment = theta((1 + (hiddenLayerSize * (numFeatures + 1))):end);
    outputTheta = reshape(outputThetaSegment, numClasses, (hiddenLayerSize + 1));
    
    %% Recode the output labels
    Y = zeros(numExamples, numClasses);
    
    for i = 1:numClasses
        Y(:, i) = y == i;
    end

    %% Calculate the cost J
    % feedforward the neural network
    inputLayer = [ones(numExamples, 1), X];
    z2 = inputLayer * hiddenTheta';
    hiddenLayer = [ones(size(z2, 1), 1), sigmoid(z2)];
    predictions = sigmoid(hiddenLayer * outputTheta');

    % compute unregularized cost
    error = -Y .* log(predictions) - (1 - Y) .* log(1 - predictions);
    
    % compute regularization penalty
    regularization = sum(sum(hiddenTheta(:, 2:end) .^ 2, 2)) + ... 
                     sum(sum(outputTheta(:, 2:end) .^ 2, 2));
       
    cost = (sum(sum(error, 2)) + lambda * regularization / 2) / numExamples;
    
    %% Backpropagation
    % calculate sigmas
    sigma3 = predictions - Y;
    sigma2 = (sigma3 * outputTheta) .* sigmoidGradient([ones(size(z2, 1), 1) z2]);
    sigma2 = sigma2(:, 2:end);

    % accumulate gradients
    hiddenDelta = sigma2' * inputLayer;
    outputDelta = sigma3' * hiddenLayer;

    % add regularization
    reg1 = lambda * [zeros(size(hiddenTheta, 1), 1), hiddenTheta(:, 2:end)];
    reg2 = lambda * [zeros(size(outputTheta, 1), 1), outputTheta(:, 2:end)];
    
    % compute and unroll the gradients
    hiddenThetaGradient = (hiddenDelta + reg1) ./ numExamples;
    outputThetaGradient = (outputDelta + reg2) ./ numExamples;
    gradient = [hiddenThetaGradient(:); outputThetaGradient(:)];
end
