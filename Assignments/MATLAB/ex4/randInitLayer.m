function W = randInitLayer(inputSize, outputSize)
% Randomly initialize the weights of a layer
    epsilon = 0.12;
    W = rand(outputSize, inputSize + 1) * 2 * epsilon - epsilon;
end
