function [hiddenTheta, outputTheta] = recoverTheta(theta, numHiddenUnits, numFeatures, numClasses)
    
    % hiddenTheta is a numHiddenUnits by (numFeatures + 1) matrix 
    hiddenThetaSegment = theta(1:numHiddenUnits * (numFeatures + 1));
    hiddenTheta = reshape(hiddenThetaSegment, numHiddenUnits, (numFeatures + 1));
    
    % outputTheta is a numClasses by (numHiddenUnits + 1) matrix
    outputThetaSegment = theta((1 + (numHiddenUnits * (numFeatures + 1))):end);
    outputTheta = reshape(outputThetaSegment, numClasses, (numHiddenUnits + 1));
end

