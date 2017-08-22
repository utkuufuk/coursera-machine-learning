function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%gradientDescent Performs gradient descent to learn theta
%   theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        
        H = X * theta;  % calculate H with the theta values calculated in the previous iteration.
                
        for i = 1:m
            
%             descent1 = (H(i) - y(i)) * X(i, 1);
%             descent2 = (H(i) - y(i)) * X(i, 2);
%             theta(1) = theta(1) - (alpha * descent1 / m);
%             theta(2) = theta(2) - (alpha * descent2 / m);

            theta(1) = theta(1) - (alpha * (H(i) - y(i)) * X(i, 1) / m);
            theta(2) = theta(2) - (alpha * (H(i) - y(i)) * X(i, 2) / m);
        end
        
        J_history(iter) = computeCost(X, y, theta);     % Save the cost J in every iteration
    end
    
%     plot(1:num_iters, J_history, '-')
end
