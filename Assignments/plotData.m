function plotData(X, y)
% Plots the data points with '+' for the positive examples
% and 'o' for the negative examples. X is assumed to be a Mx2 matrix.

    figure; hold on;
    positives = find(y == 1);
    negatives = find(y == 0);
    plot(X(positives, 1), X(positives, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot(X(negatives, 1), X(negatives, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
end
