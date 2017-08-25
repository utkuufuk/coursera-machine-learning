function predictions = predict(hiddenTheta, outputTheta, X)
%PREDICT Predict the label of an input given a trained neural network

%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the trained weights of a neural network (hiddenTheta, outputTheta)

    testSetSize = size(X, 1);    
    X = [ones(testSetSize, 1) X];   % add intercept term to X
    
    hiddenLayer = sigmoid(X * hiddenTheta');
    hiddenLayer = [ones(testSetSize, 1) hiddenLayer];   % add intercept term to hidden layer
    
    probabilities = sigmoid(hiddenLayer * outputTheta');
    [~, predictions] = max(probabilities, [], 2);
end

