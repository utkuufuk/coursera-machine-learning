function W = randInitializeWeights(inputSize, outputSize)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

    % Randomly initialize the weights to small values
    initialEpsilon = 0.12;
    W = rand(outputSize, inputSize + 1) * 2 * initialEpsilon - initialEpsilon;
end
