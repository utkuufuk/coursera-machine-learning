function [C, sigma] = dataset3Params(X, y, xVal, yVal)
%Selects the optimal (C, sigma) learning parameters to use for SVM with RBF kernel

    cIndex = 0;
    cVector = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    sigmaVector = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    validationError = zeros(numel(cVector), numel(sigmaVector));
    
    for C = cVector
        
        cIndex = cIndex + 1;
        sigmaIndex = 0;
        
        for sigma = sigmaVector
            
            sigmaIndex = sigmaIndex + 1;
            
            model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
            predictions = svmPredict(model, xVal);
            validationError(cIndex, sigmaIndex) = mean(double(predictions ~= yVal));
        end
    end
    
    [~, minIndex] = min(validationError(:));
    [cIndex, sigmaIndex] = ind2sub(size(validationError), minIndex);
    C = cVector(cIndex);
    sigma = sigmaVector(sigmaIndex);
end