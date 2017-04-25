function [theta, J_history] = gradientDescentRegularizedMulti(X, y, theta, alpha, lambda, num_iters)

    exampleCount = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iterationIndex = 1:num_iters
        
        H = X * theta;  % calculate H with the theta values calculated in the previous iteration.
        regularization = lambda * theta;
        regularization(1) = 0; % don't penalize the constant term.
        descent = (transpose(X) * (H - y) + regularization) / exampleCount;
        theta = theta - (alpha * descent);
        J_history(iterationIndex) = computeCost(X, y, theta);     % Save the cost J in every iteration
    end
end
