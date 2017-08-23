function [X, means, stDevs] = featureNormalize(X)
% Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    
    means = mean(X);
    X = X - means(ones(size(X, 1), 1), :);
    stDevs = std(X);
    X = X ./ stDevs(ones(size(X, 1), 1), :);
end
