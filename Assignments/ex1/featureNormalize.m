function [X, means, stDevs] = featureNormalize(X)
% Normalizes X where the mean of each feature is 0 and the standard deviation is 1.
    means = mean(X);
    stDevs = std(X);
    X = X - bsxfun(@minus, X, means);       % subtract the mean
    X = X - bsxfun(@rdivide, X, stDevs);    % divide by the standard deviation
end
