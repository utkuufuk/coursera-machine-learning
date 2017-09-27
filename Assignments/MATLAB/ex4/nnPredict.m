function predictions = nnPredict(hiddenTheta, outputTheta, X)
% Predicts the label of an input given a trained neural network
    X = [ones(size(X, 1), 1), X];
    hiddenLayer = [ones(size(X, 1), 1), sigmoid(X * hiddenTheta')];
    probabilities = sigmoid(hiddenLayer * outputTheta');
    [~, predictions] = max(probabilities, [], 2);
end

