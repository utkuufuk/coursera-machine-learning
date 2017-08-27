function plotFit(min_x, max_x, mu, sigma, theta, p)
% Plots the polynomial fit with power p and normalization (mu, sigma) over an existing figure

    % plot a range slightly bigger than the min and max values
    x = (min_x - 15: 0.05 : max_x + 25)';

    % map the X values 
    X_poly = polyFeatures(x, p);
    X_poly = bsxfun(@minus, X_poly, mu);
    X_poly = bsxfun(@rdivide, X_poly, sigma);

    % add ones and plot
    hold on;
    X_poly = [ones(size(x, 1), 1) X_poly];
    plot(x, X_poly * theta, '--', 'LineWidth', 2)
    hold off
end
