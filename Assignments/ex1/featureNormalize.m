function [xNorm, m, sigma] = featureNormalize(X)
% Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    
    m = mean(X);
    xNorm = X - m(ones(size(X, 1), 1), :);
    sigma = std(X);
    xNorm = xNorm ./ sigma(ones(size(X, 1), 1), :);
end
