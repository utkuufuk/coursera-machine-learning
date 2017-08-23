%% Machine Learning Online Class - Exercise 5 | Regularized Linear Regression and Bias-Variance

%% =========== Part 1: Loading and Visualizing Data =============
load('ex5data1.mat'); % You will have X, y, Xval, yval, Xtest, ytest in your environment
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

%% =========== Part 2: Regularized Linear Regression Cost and Gradient =============
theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf('Cost at theta = [1 ; 1]: %f\n(should be about 303.993192)\n', J);
fprintf('Gradient at theta = [1; 1]: [%f; %f]\n(should be about [-15.303016; 598.250744])\n', grad(1), grad(2));

%% =========== Part 3: Train Linear Regression =============
lambda = 0; % train linear regression with lambda = 0
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

%% =========== Part 4: Learning Curve for Linear Regression =============
lambda = 0;
[trainErr, valErr] = learningCurve([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, lambda);
              
%  Since the model is underfitting the data, we expect to see a graph with "high bias"
plot(1:m, trainErr, 1:m, valErr);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# of Training Examples\tTrain Error\t\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t\t%d\t\t\t\t%f\t\t%f\n', i, trainErr(i), valErr(i));
end

%% =========== Part 5: Feature Mapping for Polynomial Regression =============
% Map X onto Polynomial Features and Normalize
p = 8;
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

%% =========== Part 6: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running polynomial regression with different values of
%  lambda to see how the fit and learning curve change.

lambda = 1;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[trainErr, valErr] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, trainErr, 1:m, valErr);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, trainErr(i), valErr(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 7: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.

[lambda_vec, trainErr, valErr] = validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, trainErr, lambda_vec, valErr);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\t\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t\t%f\n', lambda_vec(i), trainErr(i), valErr(i));
end

%% =========== Part 8: Test Error with the Selected Lambda =============
lambda = 3;
theta = trainLinearReg(X_poly, y, lambda);
testError = linearRegCostFunction(X_poly_test, ytest, theta, 0)