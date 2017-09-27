function p = predictLabel(theta, X)
% Predict whether the label is 0 or 1 using logistic regression parameters
    p = zeros(size(X, 1), 1);
    p(sigmoid(X * theta) >= 0.5) = 1;
end
