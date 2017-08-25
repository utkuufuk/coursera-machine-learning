%% Machine Learning Online Class - Exercise 2a: Logistic Regression
%% ==================== Part 1: Plotting ====================
data = load('data2a.txt');
X = data(:, 1:2);   % exam scores
y = data(:, 3);     % label

plotData(X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%% ============= Part 2: Optimizing using fminunc  =============
[numExamples, numFeatures] = size(X);

% Add intercept term to x and X_test
X = [ones(numExamples, 1) X];

% Initialize fitting parameters
theta = zeros(numFeatures + 1, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(classificationCost(t, X, y, 0)), theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%% ============== Part 3: Predict and Accuracies ==============
% Predict probability for a student with scores 45 and 85
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n', prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(p == y) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');
