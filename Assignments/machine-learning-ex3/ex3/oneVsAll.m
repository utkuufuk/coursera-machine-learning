function [classifiers] = oneVsAll(X, y, numberOfClasses, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

    sampleSize = size(X, 1);
    numberOfFeatures = size(X, 2);
    
    % initialize 'numberOfClasses' classifiers
    classifiers = zeros(numberOfClasses, numberOfFeatures + 1);

    % Add intercept term to X
    X = [ones(sampleSize, 1) X];

    % train each classifier 
    for i = 1:numberOfClasses
        
        targetVariableVector = y == i;
        theta = zeros(numberOfFeatures + 1, 1);
        
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        [theta] = fmincg (@(t)(lrCostFunction(t, X, targetVariableVector, lambda)), theta, options);
        classifiers(i, 1:numberOfFeatures + 1) = theta';
    end
end
