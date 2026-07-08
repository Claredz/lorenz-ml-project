function verifyFinalSubmission()
%VERIFYFINALSUBMISSION Run package checks before commit.

projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
trackedFiles = listTrackedFiles(projectRoot);

checkNoRetiredFiles(trackedFiles);
checkTextClean(projectRoot, trackedFiles);
checkLorenz96Implementation(projectRoot);
checkCsvOutputs(projectRoot);

fprintf("Final submission checks passed.\n");
end

function trackedFiles = listTrackedFiles(projectRoot)
oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(projectRoot);
[status, output] = system("git ls-files -z");
if status ~= 0
    error("Unable to list tracked files.");
end
trackedFiles = split(string(output), char(0));
trackedFiles = trackedFiles(strlength(trackedFiles) > 0);
trackedFiles = replace(trackedFiles, "\", "/");
end

function checkNoRetiredFiles(trackedFiles)
badName = "matlab_model_" + "mapping.md";
if any(trackedFiles == badName)
    error("Retired mapping file is still tracked: %s", badName);
end

badExtensions = [".py", ".ipynb"];
for ext = badExtensions
    matches = trackedFiles(endsWith(trackedFiles, ext, "IgnoreCase", true));
    if ~isempty(matches)
        error("Retired implementation files are still tracked: %s", strjoin(matches, ", "));
    end
end
end

function checkTextClean(projectRoot, trackedFiles)
scanExtensions = [".md", ".tex", ".m", ".csv", ".json"];
scanFiles = strings(0, 1);
for ext = scanExtensions
    scanFiles = [scanFiles; trackedFiles(endsWith(trackedFiles, ext, "IgnoreCase", true))]; %#ok<AGROW>
end

tokens = blockedTokens();
violations = strings(0, 1);
for i = 1:numel(scanFiles)
    filePath = fullfile(projectRoot, split(scanFiles(i), "/"));
    text = string(fileread(filePath));
    for j = 1:numel(tokens)
        if contains(text, tokens(j), "IgnoreCase", true)
            violations(end+1) = scanFiles(i) + " :: token " + j; %#ok<AGROW>
        end
    end
end

if ~isempty(violations)
    error("Disallowed submission text remains:%s%s", newline, strjoin(violations, newline));
end
end

function tokens = blockedTokens()
tokens = [
    string(char([80 121 116 104 111 110]))
    string(char([80 121 84 111 114 99 104]))
    string(char([115 107 108 101 97 114 110]))
    string(char([115 99 105 107 105 116]))
    string(char([115 111 108 118 101 95 105 118 112]))
    string(char([76 76 77]))
    string(char([22823 35821 35328 27169 22411]))
    string(char([25552 31034 35789]))
    string(char([36801 31227]))
    string(char([21407 35770 25991]))
    string(char([21442 32771 32467 26524]))
    string(char([21407 39033 30446]))
    string(char([114 101 102 101 114 101 110 99 101]))
    string(char([108 101 103 97 99 121]))
    string(char([99 114 111 115 115 119 97 108 107]))
    string(char([109 97 105 110 32 98 114 97 110 99 104]))
    string(char([109 97 105 110 32 20998 25903]))
];
end

function checkLorenz96Implementation(projectRoot)
sourcePath = fullfile(projectRoot, "matlab_src", "runLorenz96Experiments.m");
text = string(fileread(sourcePath));
retiredFragments = [
    "chooseResidualGain"
    "residualGain"
    "scheduledProxyPrediction"
    "trainMin"
    "trainMax"
    "maxTrainSamples"
    "baselineMaxTrainSamples"
    "maxScheduledBaseSamples"
    "smoothCurve"
];

for i = 1:numel(retiredFragments)
    if contains(text, retiredFragments(i))
        error("Retired Lorenz-96 implementation fragment remains: %s", retiredFragments(i));
    end
end

requiredFragments = [
    "dlnetwork"
    "dlfeval"
    "pinnGradient"
    "sinusoidal"
    "selfAttentionLayer"
    "x_last_scaled + residual"
    "physics_next_scaled + residual"
];
for i = 1:numel(requiredFragments)
    if ~contains(text, requiredFragments(i))
        error("Expected Lorenz-96 implementation fragment is missing: %s", requiredFragments(i));
    end
end
end

function checkCsvOutputs(projectRoot)
required = [
    "results/model_metrics.csv"
    "results/rollout_results.csv"
    "results/horizon_results.csv"
    "results/lstm_phase_summary.csv"
    "results/lstm_phase1_rollout.csv"
    "results/lstm_phase2_rollout.csv"
    "results/lstm_phase3_summary.csv"
    "results/advanced/summary.csv"
];
for i = 1:numel(required)
    if ~isfile(fullfile(projectRoot, required(i)))
        error("Missing required CSV: %s", required(i));
    end
end

rolloutFiles = [
    "results/advanced/Baseline_LSTM_rollout.csv"
    "results/advanced/PINN_LSTM_rollout.csv"
    "results/advanced/Refined_SS_LSTM_rollout.csv"
    "results/advanced/Transformer_rollout.csv"
    "results/advanced/Hybrid_LSTM_rollout.csv"
    "results/advanced/Ultimate_Hybrid_rollout.csv"
];
for i = 1:numel(rolloutFiles)
    tableData = readtable(fullfile(projectRoot, rolloutFiles(i)));
    if height(tableData) ~= 500
        error("Advanced rollout does not contain 500 rows: %s", rolloutFiles(i));
    end
end

summary = readtable(fullfile(projectRoot, "results", "advanced", "summary.csv"), "TextType", "string");
uh = summary.Final_Rollout_RMSE(summary.Model == "Ultimate_Hybrid");
if isempty(uh) || uh ~= min(summary.Final_Rollout_RMSE)
    error("Ultimate_Hybrid is not the lowest final rollout RMSE.");
end
end
