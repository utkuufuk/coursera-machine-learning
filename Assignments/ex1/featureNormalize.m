function [X_norm, mu, sigma] = featureNormalize(X)
% Normalizes the features in X 
%   Returns a normalized version of X where the mean value of each feature is 0 and the 
%   standard deviation is 1. This is often a good preprocessing step to do when working 
%   with learning algorithms.
    
    m = mean(X);
    X_norm = X - m(ones(size(X, 1), 1), :);
    sigma = std(X);
    X_norm = X_norm ./ sigma(ones(size(X, 1), 1), :);
end
