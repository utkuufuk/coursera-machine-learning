function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

    J_history = zeros(num_iters, 1);

    for iterationIndex = 1:num_iters
        
        H = X * theta;  % calculate H with the theta values calculated in the previous iteration.
        descent = transpose(X) * (H - y);
        theta = theta - (alpha * descent / length(y));
        J_history(iterationIndex) = computeCost(X, y, theta);     % Save the cost J in every iteration
    end
end
