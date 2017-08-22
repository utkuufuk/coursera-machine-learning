%% Machine Learning Online Class - Exercise 1: Linear Regression
%% ======================= Part 1: Plotting =======================
data = load('ex1data1.txt');
X = data(:, 1);             % population size in 10,000s
y = data(:, 2);             % profit in $10,000s
numExamples = length(y);    % number of training examples
plotData(X, y);
%% =================== Part 2: Gradient descent ===================
X = [ones(numExamples, 1), data(:,1)];  % add a column of ones to x
theta = zeros(2, 1);                    % initialize model parameters
computeCost(X, y, theta)                % compute and display the initial cost

% Some gradient descent settings
iterations = 2400;
alpha = 0.005;

% run gradient descent
%theta = gradientDescent(X, y, theta, alpha, iterations);

[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);

% Plot the convergence graph
% figure;
% plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
% xlabel('Number of iterations');
% ylabel('Cost J');

% print theta to screen
fprintf('The final cost is: %f\n', computeCost(X, y, theta));
fprintf('Theta found by gradient descent: %f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:, 2), X * theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);
%% ============= Part 3: Visualizing J(theta_0, theta_1) =============
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
