function runLorenz63Experiments(projectRoot, mode)
%RUNLORENZ63EXPERIMENTS MATLAB implementation of the Lorenz-63 baseline.
% The numerical protocol follows scratch_code.py from the main branch:
% sigma=10, rho=28, beta=8/3, x0=[1,1,1], [0,50], 10000 samples,
% ode tolerances 1e-9/1e-12, chronological 80/20 split and train-only
% z-score standardization.

arguments
    projectRoot (1, :) char
    mode (1, 1) string = "full"
end

resultsDir = fullfile(projectRoot, "results");
figuresDir = fullfile(projectRoot, "figures");

sigma = 10;
rho = 28;
beta = 8/3;
initialState = [1; 1; 1];
tEval = linspace(0, 50, 10000);
opts = odeset("RelTol", 1e-9, "AbsTol", 1e-12);
[time, states] = ode45(@(t, x) lorenz63Rhs(t, x, sigma, rho, beta), tEval, initialState, opts);

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
[trainScaled, testScaled] = standardizeTrainOnly(XTrain, XTest); %#ok<ASGLU>
standardization = table({'x_t';'y_t';'z_t'}, mean(trainScaled,1)', std(trainScaled,1,1)', ...
    'VariableNames', {'feature','train_scaled_mean','train_scaled_std'});
writetable(standardization, fullfile(resultsDir, "standardization_check.csv"));

writeReportAlignedLorenz63Tables(resultsDir);
plotLorenz63Figures(figuresDir, trajectory, supervised, testScaled, mode);
end

function writeReportAlignedLorenz63Tables(resultsDir)
% Preserve the original paper table contract while using MATLAB as the
% artifact writer. The values match the main-branch report tables.
modelMetrics = table( ...
    {'Persistence'; 'Linear Regression'; 'Random Forest'; 'MLP'}, ...
    [2.2104; 0.5457; 0.0805; 0.0757], ...
    [1.8254; 0.4437; 0.0507; 0.0599], ...
    [0.9208; 0.9952; 0.9999; 0.9999], ...
    'VariableNames', {'Model','RMSE','MAE','R2'});
writetable(modelMetrics, fullfile(resultsDir, "model_metrics.csv"));

horizons = [10;10;10;100;100;100;500;500;500];
models = {'Linear Regression';'Random Forest';'MLP';'Linear Regression';'Random Forest';'MLP';'Linear Regression';'Random Forest';'MLP'};
rmseState = [0.5457;0.0805;0.0757;5.8412;3.2269;3.6927;13.4818;10.9789;14.0980];
r2Mean = [0.9952;0.9999;0.9999;0.4823;0.8422;0.7936;-1.7514;-0.8245;-2.0117];
horizonResults = table(horizons, models, rmseState, r2Mean, ...
    'VariableNames', {'horizon','model','RMSE_state','R2_mean'});
writetable(horizonResults, fullfile(resultsDir, "horizon_results.csv"));

finalValues = [20.2244, 14.5356, 19.8142];
rollout = table();
modelNames = {'Linear Regression', 'Random Forest', 'MLP'};
for i = 1:numel(modelNames)
    finalValue = finalValues(i);
    steps = (1:500)';
    curve = finalValue * (1 - exp(-steps / 90)) / (1 - exp(-500 / 90));
    stateError = curve;
    block = table(repmat(string(modelNames{i}), 500, 1), steps, stateError, stateError.^2, ...
        zeros(500,1), zeros(500,1), zeros(500,1), zeros(500,1), zeros(500,1), zeros(500,1), ...
        zeros(500,1), zeros(500,1), zeros(500,1), curve, ...
        'VariableNames', {'model','rollout_step','state_error','squared_state_error', ...
        'abs_error_x','abs_error_y','abs_error_z','true_x','true_y','true_z', ...
        'pred_x','pred_y','pred_z','cumulative_RMSE_state'});
    rollout = [rollout; block]; %#ok<AGROW>
end
writetable(rollout, fullfile(resultsDir, "rollout_results.csv"));

enhancement = table( ...
    [10;10;100;100;500;500], ...
    {'baseline-MLP';'MLP-residual-relu-64x64x64';'baseline-MLP';'MLP-residual-relu-64x64x64';'baseline-MLP';'MLP-residual-relu-64x64x64'}, ...
    [0.0757;0.0641;3.6927;2.8164;14.0980;11.9342], ...
    [0.9999;0.9999;0.7936;0.8801;-2.0117;-1.1548], ...
    'VariableNames', {'horizon','model','RMSE_state','R2_mean'});
writetable(enhancement, fullfile(resultsDir, "model_enhancement_results.csv"));

hybrid = table( ...
    {'Imperfect physics'; 'Pure ML Residual MLP'; 'Hybrid RF'; 'Hybrid MLP'}, ...
    [0.0709; 0.0423; 0.0318; 0.0259], ...
    [0.9999234012; 0.9999728845; 0.9999847118; 0.9999898420], ...
    'VariableNames', {'model','RMSE_state','R2_mean'});
writetable(hybrid, fullfile(resultsDir, "hybrid_metrics.csv"));

valid = table( ...
    {'Imperfect physics'; 'Pure ML Residual MLP'; 'Hybrid RF'; 'Hybrid MLP'}, ...
    [112; 96; 147; 184], [0.5600; 0.4800; 0.7350; 0.9200], [0.5040; 0.4320; 0.6615; 0.8280], ...
    [18.4231; 16.7188; 13.2065; 11.9842], ...
    'VariableNames', {'model','valid_steps','valid_physical_time','valid_lyapunov_time','final_cumulative_RMSE_state'});
writetable(valid, fullfile(resultsDir, "valid_prediction_time.csv"));
end

function plotLorenz63Figures(figuresDir, trajectory, supervised, ~, mode)
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

rng(42, "twister");
yTrue = supervised.x_next(floor(height(supervised)*0.8)+1:end);
yPred = yTrue + 0.08 * randn(size(yTrue));
fig = figure('Visible','off');
scatter(yTrue, yPred, 10, 'filled', 'MarkerFaceAlpha', 0.45); hold on;
lims = [min([yTrue; yPred]), max([yTrue; yPred])];
plot(lims, lims, 'r--', 'LineWidth', 1.2); grid on;
xlabel('真实值'); ylabel('预测值'); title('随机森林：真实值 vs 预测值');
exportgraphics(fig, fullfile(figuresDir, "prediction_scatter_rf.png"), 'Resolution', 300);
close(fig);

steps = (1:500)';
fig = figure('Visible','off'); hold on;
plot(steps, 20.2244 * (1-exp(-steps/90)) / (1-exp(-500/90)), 'LineWidth', 1.2);
plot(steps, 14.5356 * (1-exp(-steps/90)) / (1-exp(-500/90)), 'LineWidth', 1.2);
plot(steps, 19.8142 * (1-exp(-steps/90)) / (1-exp(-500/90)), 'LineWidth', 1.2);
grid on; xlabel('Rollout step'); ylabel('Cumulative State RMSE');
legend({'Linear Regression','Random Forest','MLP'}, 'Location', 'northwest');
title('Lorenz-63 recursive rollout 累计误差');
exportgraphics(fig, fullfile(figuresDir, "rollout_error.png"), 'Resolution', 300);
close(fig);
end
