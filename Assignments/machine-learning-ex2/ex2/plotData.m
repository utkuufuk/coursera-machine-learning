function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

    figure; hold on;
    positiveIndexes = find(y == 1);
    negativeIndexes = find(y == 0);
    plot(X(positiveIndexes, 1), X(positiveIndexes, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot(X(negativeIndexes, 1), X(negativeIndexes, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
    hold off;
end
