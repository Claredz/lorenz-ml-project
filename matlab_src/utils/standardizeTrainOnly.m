function [trainScaled, testScaled, mu, sigma] = standardizeTrainOnly(trainData, testData)
%STANDARDIZETRAINONLY Match Python StandardScaler using train-only statistics.
% Python sklearn.preprocessing.StandardScaler uses population standard
% deviation. MATLAB std defaults to sample normalization, so use flag 1.
mu = mean(trainData, 1);
sigma = std(trainData, 1, 1);
sigma(sigma == 0) = 1;
trainScaled = (trainData - mu) ./ sigma;
testScaled = (testData - mu) ./ sigma;
end
