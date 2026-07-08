function [trainScaled, testScaled, mu, sigma] = standardizeTrainOnly(trainData, testData)
%STANDARDIZETRAINONLY Train-only z-score scaling with population deviation.
% MATLAB std defaults to sample normalization, so use flag 1.
mu = mean(trainData, 1);
sigma = std(trainData, 1, 1);
sigma(sigma == 0) = 1;
trainScaled = (trainData - mu) ./ sigma;
testScaled = (testData - mu) ./ sigma;
end
