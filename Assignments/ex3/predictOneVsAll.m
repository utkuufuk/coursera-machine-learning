function indexes = predictOneVsAll(classifiers, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(classifiers, 1). 
%  p = PREDICTONEVSALL(classifiers, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

    testSetSize = size(X, 1);    
    X = [ones(testSetSize, 1) X];    % add intercept term to X
    predictions = sigmoid(X * classifiers');
    [~, indexes] = max(predictions, [], 2);
end
