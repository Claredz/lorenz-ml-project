function generateReportTablesMatlab(projectRoot)
%GENERATEREPORTTABLESMATLAB Generate LaTeX table fragments from MATLAB CSVs.
% This is the MATLAB counterpart of scripts/generate_report_tables.py.

resultsDir = fullfile(projectRoot, "results");
outFile = fullfile(resultsDir, "generated_report_tables.tex");
sections = strings(0, 1);
sections(end+1) = originalMetrics(resultsDir);
sections(end+1) = horizonMetrics(resultsDir);
sections(end+1) = rolloutSummary(resultsDir);
sections(end+1) = residualMetrics(resultsDir);
sections(end+1) = hybridMetrics(resultsDir);
sections(end+1) = validPredictionTime(resultsDir);
content = strjoin(sections, sprintf('\n\n'));
fid = fopen(outFile, 'w', 'n', 'UTF-8');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', content);
fprintf('Wrote %s\n', outFile);
end

function s = originalMetrics(resultsDir)
data = readtable(fullfile(resultsDir, "model_metrics.csv"), 'TextType', 'string');
models = ["Linear Regression", "Random Forest", "MLP"];
rows = strings(numel(models), 1);
for i = 1:numel(models)
    row = data(data.Model == models(i), :);
    rows(i) = sprintf('%s & %.4f & %.4f & %.4f ', models(i), row.RMSE, row.MAE, row.R2) + rowEnd();
end
s = section('Direct prediction metrics from results/model_metrics.csv', rows);
end

function s = horizonMetrics(resultsDir)
data = readtable(fullfile(resultsDir, "horizon_results.csv"), 'TextType', 'string');
wanted = [10 100 500];
models = ["Linear Regression", "Random Forest", "MLP"];
rows = strings(0, 1);
for h = wanted
    for m = models
        row = data(data.horizon == h & data.model == m, :);
        rows(end+1) = sprintf('%d & %s & %.4f & %.4f ', h, m, row.RMSE_state, row.R2_mean) + rowEnd(); %#ok<AGROW>
    end
end
s = section('Representative horizon metrics from results/horizon_results.csv', rows);
end

function s = rolloutSummary(resultsDir)
data = readtable(fullfile(resultsDir, "rollout_results.csv"), 'TextType', 'string');
models = ["Linear Regression", "Random Forest", "MLP"];
rows = strings(numel(models), 1);
for i = 1:numel(models)
    subset = data(data.model == models(i), :);
    row = subset(end, :);
    rows(i) = sprintf('%s & %d & %.4f ', models(i), row.rollout_step, row.cumulative_RMSE_state) + rowEnd();
end
s = section('Rollout final cumulative RMSE from results/rollout_results.csv', rows);
end

function s = residualMetrics(resultsDir)
data = readtable(fullfile(resultsDir, "model_enhancement_results.csv"), 'TextType', 'string');
wanted = {
    10, "baseline-MLP", "Direct MLP";
    10, "MLP-residual-relu-64x64x64", "Residual MLP";
    100, "baseline-MLP", "Direct MLP";
    100, "MLP-residual-relu-64x64x64", "Residual MLP";
    500, "baseline-MLP", "Direct MLP";
    500, "MLP-residual-relu-64x64x64", "Residual MLP"};
rows = strings(size(wanted,1), 1);
for i = 1:size(wanted,1)
    h = wanted{i,1}; key = wanted{i,2}; label = wanted{i,3};
    row = data(data.horizon == h & data.model == key, :);
    rows(i) = sprintf('%d & %s & %.4f & %.4f ', h, label, row.RMSE_state, row.R2_mean) + rowEnd();
end
s = section('Direct MLP vs Residual MLP from results/model_enhancement_results.csv', rows);
end

function s = hybridMetrics(resultsDir)
data = readtable(fullfile(resultsDir, "hybrid_metrics.csv"), 'TextType', 'string');
models = ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"];
rows = strings(numel(models), 1);
for i = 1:numel(models)
    row = data(data.model == models(i), :);
    rows(i) = sprintf('%s & %.4f & %.10f ', models(i), row.RMSE_state, row.R2_mean) + rowEnd();
end
s = section('Hybrid one-step metrics from results/hybrid_metrics.csv', rows);
end

function s = validPredictionTime(resultsDir)
data = readtable(fullfile(resultsDir, "valid_prediction_time.csv"), 'TextType', 'string');
models = ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"];
rows = strings(numel(models), 1);
for i = 1:numel(models)
    row = data(data.model == models(i), :);
    rows(i) = sprintf('%s & %d & %.4f & %.4f & %.4f ', models(i), row.valid_steps, row.valid_physical_time, row.valid_lyapunov_time, row.final_cumulative_RMSE_state) + rowEnd();
end
s = section('Valid prediction time from results/valid_prediction_time.csv', rows);
end

function s = section(title, rows)
s = "% " + title + newline + strjoin(rows, newline) + newline;
end

function e = rowEnd()
e = "\\";
end
