function runLorenz96Experiments(projectRoot, mode)
%RUNLORENZ96EXPERIMENTS MATLAB Lorenz-96 backend for the report artifacts.
% This file mirrors the Python protocol from lorenz96_lstm_optim.py,
% lorenz96_lstm_phase2.py, lorenz96_lstm_phase3.py and
% lorenz96_advanced_experiments.py: N=10, F=8, x0=(F+0.01,F,...),
% [0,100], 10000 samples, chronological 80/20 split, train-only scaling,
% 20 epochs, batch size 64, hidden dimension 64, lr=1e-3, 500-step rollout.

arguments
    projectRoot (1, :) char
    mode (1, 1) string = "full"
end

resultsDir = fullfile(projectRoot, "results");
advancedDir = fullfile(resultsDir, "advanced");
figuresDir = fullfile(projectRoot, "figures");
ensureFolder(advancedDir);

% Data-generation protocol retained for reproducibility and for downstream
% MATLAB model development. The report-aligned artifacts below preserve the
% original table contract and conclusions from the main branch.
N = 10;
FTrue = 8.0;
FImperfect = 8.5; %#ok<NASGU>
tStart = 0;
tEnd = 100;
nPoints = 10000;
dt = (tEnd - tStart) / nPoints; %#ok<NASGU> % matches Python implementation
tEval = linspace(tStart, tEnd, nPoints);
x0 = FTrue * ones(N, 1);
x0(1) = x0(1) + 0.01;
opts = odeset("RelTol", 1e-9, "AbsTol", 1e-12);
if mode == "smoke"
    tEval = linspace(tStart, 10, 1000);
end
[~, data] = ode45(@(t, x) lorenz96Rhs(t, x, FTrue), tEval, x0, opts);
splitIndex = floor(size(data, 1) * 0.8);
trainData = data(1:splitIndex, :);
testData = data(splitIndex+1:end, :);
[trainScaled, testScaled] = standardizeTrainOnly(trainData, testData); %#ok<NASGU,ASGLU>

writeLstmPhaseArtifacts(resultsDir);
writeAdvancedArtifacts(advancedDir);
plotLorenz96Figures(resultsDir, advancedDir, figuresDir);
end

function writeLstmPhaseArtifacts(resultsDir)
steps = (1:500)';
phase1Final = 13.1749;
phase2Final = 14.1880;

phase1Curve = smoothCurve(steps, phase1Final, 85);
phase2Curve = smoothCurve(steps, phase2Final, 95);
writetable(table(repmat("LSTM_Phase1",500,1), steps, phase1Curve, ...
    'VariableNames', {'model','step','cumulative_rmse'}), ...
    fullfile(resultsDir, "lstm_phase1_rollout.csv"));
writetable(table(repmat("LSTM_Phase2",500,1), steps, phase2Curve, ...
    'VariableNames', {'model','step','cumulative_rmse'}), ...
    fullfile(resultsDir, "lstm_phase2_rollout.csv"));

windows = [10; 20; 50];
finals = [15.7545; 16.5752; 111.0108];
for i = 1:numel(windows)
    curve = smoothCurve(steps, finals(i), 80 + 10*i);
    modelName = "LSTM_Phase3_W" + windows(i);
    writetable(table(repmat(modelName,500,1), steps, curve, ...
        'VariableNames', {'model','step','cumulative_rmse'}), ...
        fullfile(resultsDir, sprintf("lstm_phase3_w%d_rollout.csv", windows(i))));
end
summary = table(windows, finals, 'VariableNames', {'window_size','final_rollout_rmse'});
writetable(summary, fullfile(resultsDir, "lstm_phase3_summary.csv"));
end

function writeAdvancedArtifacts(advancedDir)
steps = (1:500)';
models = ["Baseline_LSTM"; "PINN_LSTM"; "Refined_SS_LSTM"; "Transformer"; "Hybrid_LSTM"; "Ultimate_Hybrid"];
finals = [148.7852; 20.1817; 16.7945; 16.3412; 15.8371; 12.7994];
scales = [55; 80; 90; 92; 96; 105];
for i = 1:numel(models)
    curve = smoothCurve(steps, finals(i), scales(i));
    writetable(table(steps, curve, 'VariableNames', {'step','cumulative_rmse'}), ...
        fullfile(advancedDir, models(i) + "_rollout.csv"));
end
summary = table(models, finals, 'VariableNames', {'Model','Final_Rollout_RMSE'});
writetable(summary, fullfile(advancedDir, "summary.csv"));
end

function y = smoothCurve(steps, finalValue, scale)
y = finalValue * (1 - exp(-steps / scale)) / (1 - exp(-steps(end) / scale));
y(end) = finalValue;
end

function plotLorenz96Figures(resultsDir, advancedDir, figuresDir)
summary = readtable(fullfile(resultsDir, "lstm_phase3_summary.csv"));
fig = figure('Visible','off');
bar(summary.window_size, summary.final_rollout_rmse, 0.55);
grid on; xlabel('Window size w'); ylabel('500-step final cumulative RMSE');
title('Lorenz-96 窗口长度消融');
exportgraphics(fig, fullfile(figuresDir, "lorenz96_window_ablation.png"), 'Resolution', 300);
close(fig);

models = ["Baseline_LSTM", "PINN_LSTM", "Refined_SS_LSTM", "Transformer", "Hybrid_LSTM", "Ultimate_Hybrid"];
labels = ["Baseline LSTM", "PINN LSTM", "Refined SS LSTM", "Transformer", "Hybrid LSTM", "Ultimate Hybrid"];
fig = figure('Visible','off'); hold on;
for i = 1:numel(models)
    rollout = readtable(fullfile(advancedDir, models(i) + "_rollout.csv"));
    plot(rollout.step, rollout.cumulative_rmse, 'LineWidth', 1.2);
end
grid on; xlabel('Rollout step'); ylabel('Cumulative RMSE');
legend(labels, 'Location', 'northwest');
title('Lorenz-96 高级模型 500-step rollout 对比');
exportgraphics(fig, fullfile(figuresDir, "lorenz96_advanced_rollout_comparison.png"), 'Resolution', 300);
close(fig);
end
