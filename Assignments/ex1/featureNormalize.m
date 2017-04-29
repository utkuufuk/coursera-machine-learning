function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
    
    mu = mean(X);
    X_norm = X - mu(ones(size(X, 1), 1), :);    % subtract the corresponding mean from all features
    sigma = std(X);
    X_norm = X_norm ./ sigma(ones(size(X, 1), 1), :);
end
