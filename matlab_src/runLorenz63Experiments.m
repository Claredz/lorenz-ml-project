function runLorenz63Experiments(projectRoot, mode)
%RUNLORENZ63EXPERIMENTS MATLAB implementation of the Lorenz-63 baseline.
% This function computes the CSV files from MATLAB models instead of copying
% report values: ode45 trajectory generation, chronological train/test split,
% train-only standardization, Linear Regression, bagged regression trees, MLP,
% horizon metrics, recursive rollout, and simple physics-residual hybrids.

arguments
    projectRoot (1, :) char
    mode (1, 1) string = "full"
end

resultsDir = fullfile(projectRoot, "results");
figuresDir = fullfile(projectRoot, "figures");

rng(42, "twister");
sigma = 10;
rho = 28;
beta = 8/3;
initialState = [1; 1; 1];
tEval = linspace(0, 50, 10000);
opts = odeset("RelTol", 1e-9, "AbsTol", 1e-12);
[time, states] = ode45(@(t, x) lorenz63Rhs(t, x, sigma, rho, beta), tEval, initialState, opts);
dt = time(2) - time(1);

trajectory = table(time, states(:,1), states(:,2), states(:,3), ...
    'VariableNames', {'time','x','y','z'});

predictionHorizon = 10;
supervised = table( ...
    trajectory.x(1:end-predictionHorizon), ...
    trajectory.y(1:end-predictionHorizon), ...
    trajectory.z(1:end-predictionHorizon), ...
    trajectory.x(1+predictionHorizon:end), ...
    'VariableNames', {'x_t','y_t','z_t','x_next'});
writetable(supervised, fullfile(resultsDir, "lorenz_supervised_dataset.csv"));
writeBasicDataChecks(supervised, resultsDir);

shortModels = trainShortHorizonModels(supervised, resultsDir, mode);
horizonResults = computeHorizonResults(states, mode);
writetable(horizonResults, fullfile(resultsDir, "horizon_results.csv"));

oneStepModels = trainStateModels(states, 1, mode);
rolloutResults = rolloutStateModels(oneStepModels, states, 500);
writetable(rolloutResults, fullfile(resultsDir, "rollout_results.csv"));

enhancementResults = computeEnhancementResults(states, mode);
writetable(enhancementResults, fullfile(resultsDir, "model_enhancement_results.csv"));

[hybridMetrics, validPrediction] = computeHybridResults(states, dt, mode);
writetable(hybridMetrics, fullfile(resultsDir, "hybrid_metrics.csv"));
writetable(validPrediction, fullfile(resultsDir, "valid_prediction_time.csv"));

plotLorenz63Figures(figuresDir, trajectory, supervised, shortModels, rolloutResults, mode);
end

function writeBasicDataChecks(supervised, resultsDir)
missingValues = table(supervised.Properties.VariableNames', sum(ismissing(supervised))', ...
    'VariableNames', {'feature','missing_count'});
writetable(missingValues, fullfile(resultsDir, "missing_values.csv"));

statsNames = {'x_t'; 'y_t'; 'z_t'; 'x_next'};
values = supervised{:,:};
descriptive = table(statsNames, mean(values)', std(values,0,1)', min(values)', ...
    prctile(values,25)', median(values)', prctile(values,75)', max(values)', ...
    'VariableNames', {'feature','mean','std','min','p25','p50','p75','max'});
writetable(descriptive, fullfile(resultsDir, "descriptive_stats.csv"));

splitIndex = floor(height(supervised) * 0.8);
X = supervised{:, {'x_t','y_t','z_t'}};
XTrain = X(1:splitIndex, :);
XTest = X(splitIndex+1:end, :);
[trainScaled, ~] = standardizeTrainOnly(XTrain, XTest);
standardization = table({'x_t';'y_t';'z_t'}, mean(trainScaled,1)', std(trainScaled,1,1)', ...
    'VariableNames', {'feature','train_scaled_mean','train_scaled_std'});
writetable(standardization, fullfile(resultsDir, "standardization_check.csv"));
end

function models = trainShortHorizonModels(supervised, resultsDir, mode)
X = supervised{:, {'x_t','y_t','z_t'}};
y = supervised.x_next;
splitIndex = floor(height(supervised) * 0.8);
XTrain = X(1:splitIndex, :);
XTest = X(splitIndex+1:end, :);
yTrain = y(1:splitIndex, :);
yTest = y(splitIndex+1:end, :);
[XTrainScaled, XTestScaled, mu, scale] = standardizeTrainOnly(XTrain, XTest);

models = struct();
models.yTest = yTest;
models.persistencePred = XTest(:, 1);
models.linear = trainLinearModel(XTrainScaled, yTrain);
models.linearPred = predictLinearModel(models.linear, XTestScaled);
models.rf = trainTreeModel(XTrainScaled, yTrain, treeCount(mode, 300));
models.rfPred = predictTreeModel(models.rf, XTestScaled);
models.mlp = trainMlpModel(XTrainScaled, yTrain, mode);
models.mlpPred = predictMlpModel(models.mlp, XTestScaled);
models.mu = mu;
models.scale = scale;

modelNames = ["Persistence"; "Linear Regression"; "Random Forest"; "MLP"];
preds = {models.persistencePred, models.linearPred, models.rfPred, models.mlpPred};
metrics = zeros(numel(modelNames), 3);
for i = 1:numel(modelNames)
    metrics(i, :) = regressionMetrics(yTest, preds{i});
end
modelMetrics = table(modelNames, metrics(:,1), metrics(:,2), metrics(:,3), ...
    'VariableNames', {'Model','RMSE','MAE','R2'});
writetable(modelMetrics, fullfile(resultsDir, "model_metrics.csv"));
end

function results = computeHorizonResults(states, mode)
horizons = [10, 100, 500];
modelNames = ["Linear Regression", "Random Forest", "MLP"];
results = table();
for horizon = horizons
    models = trainStateModels(states, horizon, mode);
    XTest = models.XTestScaled;
    YTest = models.YTest;
    predictions = {
        predictLinearModel(models.linear, XTest), ...
        predictStateTreeModels(models.rf, XTest), ...
        predictStateMlpModels(models.mlp, XTest)};
    for i = 1:numel(modelNames)
        [rmseState, maeState, r2Mean, axisMetrics] = stateMetrics(YTest, predictions{i}); %#ok<ASGLU>
        row = table(horizon, modelNames(i), rmseState, maeState, r2Mean, ...
            axisMetrics(1,1), axisMetrics(1,2), axisMetrics(1,3), ...
            axisMetrics(2,1), axisMetrics(2,2), axisMetrics(2,3), ...
            axisMetrics(3,1), axisMetrics(3,2), axisMetrics(3,3), ...
            'VariableNames', {'horizon','model','RMSE_state','MAE_state','R2_mean', ...
            'RMSE_x','MAE_x','R2_x','RMSE_y','MAE_y','R2_y','RMSE_z','MAE_z','R2_z'});
        results = [results; row]; %#ok<AGROW>
    end
end
end

function models = trainStateModels(states, horizon, mode)
X = states(1:end-horizon, :);
Y = states(1+horizon:end, :);
splitIndex = floor(size(X, 1) * 0.8);
XTrain = X(1:splitIndex, :);
XTest = X(splitIndex+1:end, :);
YTrain = Y(1:splitIndex, :);
YTest = Y(splitIndex+1:end, :);
[XTrainScaled, XTestScaled, mu, scale] = standardizeTrainOnly(XTrain, XTest);

models = struct();
models.horizon = horizon;
models.mu = mu;
models.scale = scale;
models.XTestScaled = XTestScaled;
models.YTest = YTest;
models.linear = trainLinearModel(XTrainScaled, YTrain);
models.rf = trainStateTreeModels(XTrainScaled, YTrain, treeCount(mode, 120));
models.mlp = trainStateMlpModels(XTrainScaled, YTrain, mode);
end

function results = rolloutStateModels(models, states, rolloutSteps)
splitIndex = floor(size(states, 1) * 0.8);
testStates = states(splitIndex+1:end, :);
initialState = testStates(1, :);
trueRollout = testStates(2:rolloutSteps+1, :);
modelNames = ["Linear Regression", "Random Forest", "MLP"];
results = table();
for name = modelNames
    current = initialState;
    preds = zeros(rolloutSteps, 3);
    for step = 1:rolloutSteps
        currentScaled = (current - models.mu) ./ models.scale;
        switch name
            case "Linear Regression"
                nextState = predictLinearModel(models.linear, currentScaled);
            case "Random Forest"
                nextState = predictStateTreeModels(models.rf, currentScaled);
            case "MLP"
                nextState = predictStateMlpModels(models.mlp, currentScaled);
        end
        preds(step, :) = nextState;
        current = nextState;
    end
    diff = trueRollout - preds;
    axisAbs = abs(diff);
    stateError = vecnorm(diff, 2, 2);
    cumulative = cumulativeRmse(stateError);
    block = table(repmat(name, rolloutSteps, 1), (1:rolloutSteps)', stateError, stateError.^2, ...
        axisAbs(:,1), axisAbs(:,2), axisAbs(:,3), ...
        trueRollout(:,1), trueRollout(:,2), trueRollout(:,3), ...
        preds(:,1), preds(:,2), preds(:,3), cumulative, ...
        'VariableNames', {'model','rollout_step','state_error','squared_state_error', ...
        'abs_error_x','abs_error_y','abs_error_z','true_x','true_y','true_z', ...
        'pred_x','pred_y','pred_z','cumulative_RMSE_state'});
    results = [results; block]; %#ok<AGROW>
end
end

function enhancementResults = computeEnhancementResults(states, mode)
horizons = [10, 100, 500];
enhancementResults = table();
for horizon = horizons
    X = states(1:end-horizon, :);
    Y = states(1+horizon:end, :);
    splitIndex = floor(size(X, 1) * 0.8);
    XTrain = X(1:splitIndex, :);
    XTest = X(splitIndex+1:end, :);
    YTrain = Y(1:splitIndex, :);
    YTest = Y(splitIndex+1:end, :);
    [XTrainScaled, XTestScaled] = standardizeTrainOnly(XTrain, XTest);

    baselineMlp = trainStateMlpModels(XTrainScaled, YTrain, mode);
    baselinePred = predictStateMlpModels(baselineMlp, XTestScaled);
    residualMlp = trainStateMlpModels(XTrainScaled, YTrain - XTrain, mode);
    residualPred = XTest + predictStateMlpModels(residualMlp, XTestScaled);

    labels = ["baseline-MLP", "MLP-residual-relu-64x64x64"];
    preds = {baselinePred, residualPred};
    for i = 1:numel(labels)
        [rmseState, maeState, r2Mean, axisMetrics] = stateMetrics(YTest, preds{i}); %#ok<ASGLU>
        row = table(horizon, labels(i), rmseState, maeState, r2Mean, ...
            axisMetrics(1,1), axisMetrics(1,2), axisMetrics(1,3), ...
            axisMetrics(2,1), axisMetrics(2,2), axisMetrics(2,3), ...
            axisMetrics(3,1), axisMetrics(3,2), axisMetrics(3,3), ...
            'VariableNames', {'horizon','model','RMSE_state','MAE_state','R2_mean', ...
            'RMSE_x','MAE_x','R2_x','RMSE_y','MAE_y','R2_y','RMSE_z','MAE_z','R2_z'});
        enhancementResults = [enhancementResults; row]; %#ok<AGROW>
    end
end
end

function [hybridMetrics, validPrediction] = computeHybridResults(states, dt, mode)
rhoImperfect = 26.0;
rolloutSteps = 500;
X = states(1:end-1, :);
Y = states(2:end, :);
splitIndex = floor(size(X, 1) * 0.8);
XTrain = X(1:splitIndex, :);
XTest = X(splitIndex+1:end, :);
YTrain = Y(1:splitIndex, :);
YTest = Y(splitIndex+1:end, :);
[XTrainScaled, XTestScaled, mu, scale] = standardizeTrainOnly(XTrain, XTest);

physicsTrain = imperfectLorenz63StepBatch(XTrain, dt, rhoImperfect);
physicsTest = imperfectLorenz63StepBatch(XTest, dt, rhoImperfect);
residualTrain = YTrain - physicsTrain;
pureResidualTrain = YTrain - XTrain;

pureMlp = trainStateMlpModels(XTrainScaled, pureResidualTrain, mode);
hybridRf = trainStateTreeModels(XTrainScaled, residualTrain, treeCount(mode, 120));
hybridMlp = trainStateMlpModels(XTrainScaled, residualTrain, mode);

preds = {
    physicsTest, ...
    XTest + predictStateMlpModels(pureMlp, XTestScaled), ...
    physicsTest + predictStateTreeModels(hybridRf, XTestScaled), ...
    physicsTest + predictStateMlpModels(hybridMlp, XTestScaled)};
labels = ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"];
hybridMetrics = table();
for i = 1:numel(labels)
    [rmseState, maeState, r2Mean, axisMetrics] = stateMetrics(YTest, preds{i}); %#ok<ASGLU>
    row = table(labels(i), rmseState, maeState, r2Mean, ...
        axisMetrics(1,1), axisMetrics(1,2), axisMetrics(1,3), ...
        axisMetrics(2,1), axisMetrics(2,2), axisMetrics(2,3), ...
        axisMetrics(3,1), axisMetrics(3,2), axisMetrics(3,3), ...
        'VariableNames', {'model','RMSE_state','MAE_state','R2_mean', ...
        'RMSE_x','MAE_x','R2_x','RMSE_y','MAE_y','R2_y','RMSE_z','MAE_z','R2_z'});
    hybridMetrics = [hybridMetrics; row]; %#ok<AGROW>
end

models = {"physics", pureMlp, hybridRf, hybridMlp};
testStates = states(floor(size(states,1)*0.8)+1:end, :);
trueRollout = testStates(2:rolloutSteps+1, :);
threshold = 0.4 * norm(std(XTrain, 1, 1));
validPrediction = table();
for i = 1:numel(labels)
    current = testStates(1, :);
    predRollout = zeros(rolloutSteps, 3);
    for step = 1:rolloutSteps
        currentScaled = (current - mu) ./ scale;
        switch labels(i)
            case "Imperfect physics"
                nextState = imperfectLorenz63Step(current, dt, rhoImperfect);
            case "Pure ML Residual MLP"
                nextState = current + predictStateMlpModels(models{i}, currentScaled);
            case "Hybrid RF"
                nextState = imperfectLorenz63Step(current, dt, rhoImperfect) + predictStateTreeModels(models{i}, currentScaled);
            case "Hybrid MLP"
                nextState = imperfectLorenz63Step(current, dt, rhoImperfect) + predictStateMlpModels(models{i}, currentScaled);
        end
        predRollout(step, :) = nextState;
        current = nextState;
    end
    errors = vecnorm(trueRollout - predRollout, 2, 2);
    failure = find(errors > threshold, 1, 'first');
    if isempty(failure)
        validSteps = rolloutSteps;
    else
        validSteps = failure - 1;
    end
    cumulative = cumulativeRmse(errors);
    row = table(labels(i), validSteps, validSteps * dt, validSteps * dt * 0.9, cumulative(end), ...
        'VariableNames', {'model','valid_steps','valid_physical_time','valid_lyapunov_time','final_cumulative_RMSE_state'});
    validPrediction = [validPrediction; row]; %#ok<AGROW>
end
end

function plotLorenz63Figures(figuresDir, trajectory, supervised, shortModels, rolloutResults, mode)
if mode == "smoke"
    plotStride = 20;
else
    plotStride = 1;
end

fig = figure('Visible','off');
plot3(trajectory.x(1:plotStride:end), trajectory.y(1:plotStride:end), trajectory.z(1:plotStride:end), 'LineWidth', 0.4);
grid on; xlabel('x'); ylabel('y'); zlabel('z'); title('Lorenz-63 三维吸引子');
exportgraphics(fig, fullfile(figuresDir, "lorenz_attractor.png"), 'Resolution', 300);
close(fig);

corrMatrix = corr(supervised{:,:});
fig = figure('Visible','off');
imagesc(corrMatrix); axis square; colorbar; colormap(parula);
xticks(1:4); yticks(1:4); xticklabels(supervised.Properties.VariableNames); yticklabels(supervised.Properties.VariableNames);
title('Lorenz-63 监督学习数据相关性热力图');
exportgraphics(fig, fullfile(figuresDir, "correlation_heatmap.png"), 'Resolution', 300);
close(fig);

fig = figure('Visible','off');
scatter(shortModels.yTest, shortModels.rfPred, 10, 'filled', 'MarkerFaceAlpha', 0.45); hold on;
lims = [min([shortModels.yTest; shortModels.rfPred]), max([shortModels.yTest; shortModels.rfPred])];
plot(lims, lims, 'r--', 'LineWidth', 1.2); grid on;
xlabel('真实值'); ylabel('预测值'); title('随机森林：真实值 vs 预测值');
exportgraphics(fig, fullfile(figuresDir, "prediction_scatter_rf.png"), 'Resolution', 300);
close(fig);

fig = figure('Visible','off'); hold on;
modelNames = ["Linear Regression", "Random Forest", "MLP"];
for i = 1:numel(modelNames)
    subset = rolloutResults(rolloutResults.model == modelNames(i), :);
    plot(subset.rollout_step, subset.cumulative_RMSE_state, 'LineWidth', 1.2);
end
grid on; xlabel('Rollout step'); ylabel('Cumulative State RMSE');
legend(cellstr(modelNames), 'Location', 'northwest');
title('Lorenz-63 recursive rollout 累计误差');
exportgraphics(fig, fullfile(figuresDir, "rollout_error.png"), 'Resolution', 300);
close(fig);
end

function model = trainLinearModel(X, Y)
model = [ones(size(X, 1), 1), X] \ Y;
end

function y = predictLinearModel(model, X)
y = [ones(size(X, 1), 1), X] * model;
end

function model = trainTreeModel(X, y, nTrees)
model = TreeBagger(nTrees, X, y, 'Method', 'regression', 'MinLeafSize', 2, 'OOBPrediction', 'off');
end

function y = predictTreeModel(model, X)
y = predict(model, X);
if iscell(y)
    y = str2double(y);
end
y = double(y);
end

function models = trainStateTreeModels(X, Y, nTrees)
models = cell(1, size(Y, 2));
for j = 1:size(Y, 2)
    models{j} = trainTreeModel(X, Y(:, j), nTrees);
end
end

function Y = predictStateTreeModels(models, X)
Y = zeros(size(X, 1), numel(models));
for j = 1:numel(models)
    Y(:, j) = predictTreeModel(models{j}, X);
end
end

function model = trainMlpModel(X, y, mode)
iterations = 600;
if mode == "smoke"
    iterations = 60;
end
model = fitrnet(X, y, 'LayerSizes', [64 64], 'Activations', 'relu', ...
    'Lambda', 1e-4, 'Standardize', false, 'IterationLimit', iterations);
end

function y = predictMlpModel(model, X)
y = predict(model, X);
y = double(y);
end

function models = trainStateMlpModels(X, Y, mode)
models = cell(1, size(Y, 2));
for j = 1:size(Y, 2)
    models{j} = trainMlpModel(X, Y(:, j), mode);
end
end

function Y = predictStateMlpModels(models, X)
Y = zeros(size(X, 1), numel(models));
for j = 1:numel(models)
    Y(:, j) = predictMlpModel(models{j}, X);
end
end

function metrics = regressionMetrics(yTrue, yPred)
yTrue = yTrue(:);
yPred = yPred(:);
rmse = sqrt(mean((yTrue - yPred).^2));
mae = mean(abs(yTrue - yPred));
r2 = 1 - sum((yTrue - yPred).^2) / sum((yTrue - mean(yTrue)).^2);
metrics = [rmse, mae, r2];
end

function [rmseState, maeState, r2Mean, axisMetrics] = stateMetrics(YTrue, YPred)
diff = YTrue - YPred;
rmseState = sqrt(mean(sum(diff.^2, 2)));
maeState = mean(vecnorm(abs(diff), 1, 2));
axisMetrics = zeros(size(YTrue, 2), 3);
for j = 1:size(YTrue, 2)
    axisMetrics(j, :) = regressionMetrics(YTrue(:, j), YPred(:, j));
end
r2Mean = mean(axisMetrics(:, 3));
end

function count = treeCount(mode, fullCount)
if mode == "smoke"
    count = min(25, fullCount);
else
    count = fullCount;
end
end

function next = imperfectLorenz63StepBatch(X, dt, rhoImperfect)
next = zeros(size(X));
for i = 1:size(X, 1)
    next(i, :) = imperfectLorenz63Step(X(i, :), dt, rhoImperfect);
end
end

function next = imperfectLorenz63Step(x, dt, rhoImperfect)
x = x(:);
sigma = 10;
beta = 8/3;
f = @(state) [
    sigma * (state(2) - state(1));
    state(1) * (rhoImperfect - state(3)) - state(2);
    state(1) * state(2) - beta * state(3)];
k1 = f(x);
k2 = f(x + 0.5 * dt * k1);
k3 = f(x + 0.5 * dt * k2);
k4 = f(x + dt * k3);
next = (x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)).';
end
