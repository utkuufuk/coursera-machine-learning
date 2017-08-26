function W = randInitLayer(inputSize, outputSize)
% Randomly initialize the weights of a layer
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

    epsilon = 0.12;
    W = rand(outputSize, inputSize + 1) * 2 * epsilon - epsilon;
end
