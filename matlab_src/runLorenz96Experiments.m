function runLorenz96Experiments(projectRoot, mode)
%RUNLORENZ96EXPERIMENTS Train MATLAB Lorenz-96 models for the paper.
% The experiment uses ode45 data generation, chronological train/test split,
% train-only z-score scaling, neural network training, and recursive rollout.

arguments
    projectRoot (1, :) char
    mode (1, 1) string = "full"
end

cfg = lorenz96Config(mode);
resultsDir = fullfile(projectRoot, "results");
advancedDir = fullfile(resultsDir, "advanced");
figuresDir = fullfile(projectRoot, "figures");
ensureFolder(advancedDir);

rng(42, "twister");
data = generateLorenz96Data(cfg);
splitIndex = floor(size(data, 1) * 0.8);
trainData = data(1:splitIndex, :);
testData = data(splitIndex+1:end, :);
[trainScaled, testScaled, mu, sigma] = standardizeTrainOnly(trainData, testData);
ctx = struct('trainData', trainData, 'testData', testData, 'trainScaled', trainScaled, ...
    'testScaled', testScaled, 'mu', mu, 'sigma', sigma, 'cfg', cfg);

fprintf("Training Lorenz-96 phase models in MATLAB...\n");
phaseSummary = trainPhaseModels(ctx, resultsDir);
writetable(phaseSummary, fullfile(resultsDir, "lstm_phase_summary.csv"));

fprintf("Training Lorenz-96 advanced models in MATLAB...\n");
advancedSummary = trainAdvancedModels(ctx, advancedDir);
writetable(advancedSummary, fullfile(advancedDir, "summary.csv"));

plotLorenz96Figures(resultsDir, advancedDir, figuresDir);
writeRunStatus(resultsDir);
end

function cfg = lorenz96Config(mode)
cfg = struct();
cfg.mode = mode;
cfg.N = 10;
cfg.FTrue = 8.0;
cfg.FImperfect = 8.5;
cfg.tStart = 0;
cfg.tEnd = 100;
cfg.nPoints = 10000;
cfg.dt = (cfg.tEnd - cfg.tStart) / cfg.nPoints;
cfg.windowSize = 20;
cfg.epochs = 20;
cfg.batchSize = 64;
cfg.hiddenDim = 64;
cfg.learnRate = 1e-3;
cfg.mSteps = 5;
cfg.gradientThreshold = 1.0;
cfg.pinnLambda = 0.1;
cfg.rolloutSteps = 500;
cfg.smokeTrainLimit = Inf;
cfg.smokeScheduledLimit = Inf;
if mode == "smoke"
    cfg.tEnd = 30;
    cfg.nPoints = 3000;
    cfg.epochs = 2;
    cfg.rolloutSteps = 100;
    cfg.smokeTrainLimit = 350;
    cfg.smokeScheduledLimit = 160;
end
end

function data = generateLorenz96Data(cfg)
tEval = linspace(cfg.tStart, cfg.tEnd, cfg.nPoints);
x0 = cfg.FTrue * ones(cfg.N, 1);
x0(1) = x0(1) + 0.01;
opts = odeset("RelTol", 1e-9, "AbsTol", 1e-12);
[~, data] = ode45(@(t, x) lorenz96Rhs(t, x, cfg.FTrue), tEval, x0, opts);
end

function phaseSummary = trainPhaseModels(ctx, resultsDir)
cfg = ctx.cfg;
phaseRows = table();

baseline = trainSequenceModel(ctx, cfg.windowSize, "baselineLstm", "absolute", false, false, false, "none");
[baselineRollout, baselineOneStep] = evaluateSequenceModel(baseline, ctx, cfg.windowSize, "Baseline_LSTM");
writetable(withModelColumn(baselineRollout, "Baseline_LSTM"), fullfile(resultsDir, "lstm_baseline_rollout.csv"));
phaseRows = [phaseRows; phaseRow("Baseline LSTM（绝对状态预测）", baselineOneStep, baselineRollout.cumulative_rmse(end))]; %#ok<AGROW>

phase1 = trainSequenceModel(ctx, cfg.windowSize, "lstm", "residual", false, false, false, "none");
[phase1Rollout, phase1OneStep] = evaluateSequenceModel(phase1, ctx, cfg.windowSize, "LSTM_Phase1");
writetable(withModelColumn(phase1Rollout, "LSTM_Phase1"), fullfile(resultsDir, "lstm_phase1_rollout.csv"));
phaseRows = [phaseRows; phaseRow("Phase 1：残差预测 + LayerNorm + 裁剪", phase1OneStep, phase1Rollout.cumulative_rmse(end))]; %#ok<AGROW>

phase2 = trainSequenceModel(ctx, cfg.windowSize, "lstm", "residual", false, false, true, "linear");
[phase2Rollout, phase2OneStep] = evaluateSequenceModel(phase2, ctx, cfg.windowSize, "LSTM_Phase2");
writetable(withModelColumn(phase2Rollout, "LSTM_Phase2"), fullfile(resultsDir, "lstm_phase2_rollout.csv"));
phaseRows = [phaseRows; phaseRow("Phase 2：多步损失 + Scheduled Sampling", phase2OneStep, phase2Rollout.cumulative_rmse(end))]; %#ok<AGROW>

windows = [10; 20; 50];
windowSummary = table();
for i = 1:numel(windows)
    windowModel = trainSequenceModel(ctx, windows(i), "lstm", "residual", false, false, false, "none");
    [rollout, ~] = evaluateSequenceModel(windowModel, ctx, windows(i), "LSTM_Phase3_W" + windows(i));
    writetable(withModelColumn(rollout, "LSTM_Phase3_W" + windows(i)), ...
        fullfile(resultsDir, sprintf("lstm_phase3_w%d_rollout.csv", windows(i))));
    windowSummary = [windowSummary; table(windows(i), rollout.cumulative_rmse(end), ...
        'VariableNames', {'window_size','final_rollout_rmse'})]; %#ok<AGROW>
end
writetable(windowSummary, fullfile(resultsDir, "lstm_phase3_summary.csv"));
phaseSummary = phaseRows;
end

function advancedSummary = trainAdvancedModels(ctx, advancedDir)
cfg = ctx.cfg;
modelSpecs = [
    struct('name', "PINN_LSTM", 'arch', "lstm", 'target', "residual", 'hybrid', false, 'pinn', true,  'scheduled', false, 'schedule', "none")
    struct('name', "Refined_SS_LSTM", 'arch', "lstm", 'target', "residual", 'hybrid', false, 'pinn', false, 'scheduled', true,  'schedule', "refined")
    struct('name', "Transformer", 'arch', "attention", 'target', "residual", 'hybrid', false, 'pinn', false, 'scheduled', false, 'schedule', "none")
    struct('name', "Hybrid_LSTM", 'arch', "lstm", 'target', "hybridResidual", 'hybrid', true,  'pinn', false, 'scheduled', false, 'schedule', "none")
    struct('name', "Ultimate_Hybrid", 'arch', "attention", 'target', "hybridResidual", 'hybrid', true,  'pinn', true,  'scheduled', true,  'schedule', "refined")
];

baseline = trainSequenceModel(ctx, cfg.windowSize, "baselineLstm", "absolute", false, false, false, "none");
[baselineRollout, ~] = evaluateSequenceModel(baseline, ctx, cfg.windowSize, "Baseline_LSTM");
writetable(baselineRollout, fullfile(advancedDir, "Baseline_LSTM_rollout.csv"));
advancedSummary = table("Baseline_LSTM", baselineRollout.cumulative_rmse(end), ...
    'VariableNames', {'Model','Final_Rollout_RMSE'});

for i = 1:numel(modelSpecs)
    spec = modelSpecs(i);
    model = trainSequenceModel(ctx, cfg.windowSize, spec.arch, spec.target, spec.hybrid, spec.pinn, spec.scheduled, spec.schedule);
    [rollout, ~] = evaluateSequenceModel(model, ctx, cfg.windowSize, spec.name);
    writetable(rollout, fullfile(advancedDir, spec.name + "_rollout.csv"));
    advancedSummary = [advancedSummary; table(spec.name, rollout.cumulative_rmse(end), ...
        'VariableNames', {'Model','Final_Rollout_RMSE'})]; %#ok<AGROW>
end
end

function model = trainSequenceModel(ctx, windowSize, arch, targetKind, useHybrid, usePinn, useScheduled, scheduleKind)
if usePinn || useScheduled
    model = trainCustomSequenceModel(ctx, windowSize, arch, targetKind, useHybrid, usePinn, useScheduled, scheduleKind);
    return;
end

net = trainOneStepNetwork(ctx, windowSize, arch, targetKind, useHybrid, ctx.cfg.epochs);
model = modelStruct(net, false, arch, targetKind, useHybrid, usePinn, useScheduled, windowSize, ctx);
end

function model = trainCustomSequenceModel(ctx, windowSize, arch, targetKind, useHybrid, usePinn, useScheduled, scheduleKind)
cfg = ctx.cfg;
pretrainEpochs = max(1, ceil(cfg.epochs / 4));
pretrainedNet = trainOneStepNetwork(ctx, windowSize, arch, targetKind, useHybrid, pretrainEpochs);
net = dag2dlnetwork(pretrainedNet);
stepsPerBase = 1;
sampleLimit = cfg.smokeTrainLimit;
if useScheduled
    stepsPerBase = cfg.mSteps;
    sampleLimit = cfg.smokeScheduledLimit;
end
availableBase = size(ctx.trainScaled, 1) - windowSize - stepsPerBase + 1;
baseIndices = sampleIndices(availableBase, sampleLimit);

trailingAvg = [];
trailingAvgSq = [];
iteration = 0;
for epoch = 1:cfg.epochs
    baseIndices = baseIndices(randperm(numel(baseIndices)));
    pTeacher = teacherForcingProbability(epoch, cfg.epochs, scheduleKind);
    for startIdx = 1:cfg.batchSize:numel(baseIndices)
        batchIndices = baseIndices(startIdx:min(startIdx + cfg.batchSize - 1, numel(baseIndices)));
        teacherMask = rand(max(stepsPerBase - 1, 0), numel(batchIndices), 'single') < pTeacher;
        iteration = iteration + 1;
        [loss, gradients] = dlfeval(@pinnGradient, net, ctx, batchIndices, windowSize, ...
            targetKind, useHybrid, usePinn, useScheduled, stepsPerBase, teacherMask);
        gradients = clipGradients(gradients, cfg.gradientThreshold);
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, trailingAvg, trailingAvgSq, iteration, cfg.learnRate);
    end
    fprintf("  %s/%s epoch %d/%d, loss %.4g, teacher forcing %.3f\n", arch, targetKind, ...
        epoch, cfg.epochs, double(gather(extractdata(loss))), pTeacher);
end

model = modelStruct(net, true, arch, targetKind, useHybrid, usePinn, useScheduled, windowSize, ctx);
end

function net = trainOneStepNetwork(ctx, windowSize, arch, targetKind, useHybrid, epochs)
[XTrain, YTrain] = makeOneStepDataset(ctx, windowSize, targetKind, useHybrid);
layers = buildNetworkGraph(ctx.cfg.N, ctx.cfg.hiddenDim, arch, true);
options = trainingOptions('adam', ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', ctx.cfg.batchSize, ...
    'InitialLearnRate', ctx.cfg.learnRate, ...
    'GradientThreshold', ctx.cfg.gradientThreshold, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);
net = trainNetwork(XTrain, YTrain, layers, options);
end

function [loss, gradients] = pinnGradient(net, ctx, batchIndices, windowSize, targetKind, ...
    useHybrid, usePinn, useScheduled, stepsPerBase, teacherMask)
cfg = ctx.cfg;
currentSeq = dlarray(makeDlSequenceBatch(ctx.trainScaled, batchIndices, windowSize), "CBT");
dataLoss = dlarray(single(0));
pinnLoss = dlarray(single(0));

for step = 1:stepsPerBase
    trueNext = dlarray(makeDlFutureBatch(ctx.trainScaled, batchIndices, windowSize, step), "CB");
    raw = forward(net, currentSeq);
    predNext = composePredictionDl(raw, currentSeq, ctx, targetKind, useHybrid);
    dataLoss = dataLoss + mean((predNext - trueNext) .^ 2, 'all');

    if usePinn
        prevScaled = squeeze(currentSeq(:, :, end));
        predUnscaled = unscaleDataDl(predNext, ctx.mu, ctx.sigma);
        prevUnscaled = unscaleDataDl(prevScaled, ctx.mu, ctx.sigma);
        dfApprox = (predUnscaled - prevUnscaled) / cfg.dt;
        dfPhysics = lorenz96RhsBatchDl(prevUnscaled, cfg.FTrue);
        pinnLoss = pinnLoss + mean((dfApprox - dfPhysics) .^ 2, 'all');
    end

    if useScheduled && step < stepsPerBase
        mask = dlarray(reshape(teacherMask(step, :), 1, numel(batchIndices), 1), "CBT");
        trueNext3 = dlarray(reshape(stripdims(trueNext), cfg.N, numel(batchIndices), 1), "CBT");
        predNext3 = dlarray(reshape(stripdims(predNext), cfg.N, numel(batchIndices), 1), "CBT");
        nextInput = mask .* trueNext3 + (1 - mask) .* predNext3;
        currentSeq = cat(3, currentSeq(:, :, 2:end), nextInput);
    end
end

dataLoss = dataLoss / stepsPerBase;
pinnLoss = pinnLoss / stepsPerBase;
if usePinn
    loss = dataLoss + cfg.pinnLambda * pinnLoss;
else
    loss = dataLoss;
end
gradients = dlgradient(loss, net.Learnables);
end

function layers = buildNetworkGraph(N, hiddenDim, arch, includeRegression)
switch arch
    case "baselineLstm"
        layers = [
            sequenceInputLayer(N, 'Name', 'input')
            lstmLayer(hiddenDim, 'OutputMode', 'last', 'Name', 'lstm')
            fullyConnectedLayer(N, 'Name', 'fc')];
        if includeRegression
            layers = [layers; regressionLayer('Name', 'regression')];
        end
    case "lstm"
        layers = [
            sequenceInputLayer(N, 'Name', 'input')
            lstmLayer(hiddenDim, 'OutputMode', 'last', 'Name', 'lstm')
            layerNormalizationLayer('Name', 'layer_norm')
            fullyConnectedLayer(N, 'Name', 'fc')];
        if includeRegression
            layers = [layers; regressionLayer('Name', 'regression')];
        end
    case "attention"
        layers = buildTransformerGraph(N, hiddenDim, includeRegression);
    otherwise
        error("Unknown architecture: %s", arch);
end
end

function lgraph = buildTransformerGraph(N, hiddenDim, includeRegression)
ffnDim = 128;
lgraph = layerGraph();
lgraph = addLayers(lgraph, [
    sequenceInputLayer(N, 'Name', 'input')
    fullyConnectedLayer(hiddenDim, 'Name', 'embedding')]);
lgraph = addLayers(lgraph, sinusoidalPositionEncodingLayer(hiddenDim, 'Name', 'sinusoidal_pos'));
lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'add_pos'));

for block = 1:2
    prefix = "enc" + block;
    lgraph = addLayers(lgraph, selfAttentionLayer(4, hiddenDim, 'Name', prefix + "_self_attention"));
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', prefix + "_add_attention"));
    lgraph = addLayers(lgraph, layerNormalizationLayer('Name', prefix + "_ln_attention"));
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(ffnDim, 'Name', prefix + "_ffn_expand")
        reluLayer('Name', prefix + "_relu")
        fullyConnectedLayer(hiddenDim, 'Name', prefix + "_ffn_project")]);
    lgraph = addLayers(lgraph, additionLayer(2, 'Name', prefix + "_add_ffn"));
    lgraph = addLayers(lgraph, layerNormalizationLayer('Name', prefix + "_ln_ffn"));
end

lgraph = addLayers(lgraph, [
    indexing1dLayer('last', 'Name', 'last_step')
    fullyConnectedLayer(N, 'Name', 'fc')]);
if includeRegression
    lgraph = addLayers(lgraph, regressionLayer('Name', 'regression'));
end

lgraph = connectLayers(lgraph, 'embedding', 'sinusoidal_pos');
lgraph = connectLayers(lgraph, 'embedding', 'add_pos/in1');
lgraph = connectLayers(lgraph, 'sinusoidal_pos', 'add_pos/in2');
previous = "add_pos";
for block = 1:2
    prefix = "enc" + block;
    lgraph = connectLayers(lgraph, previous, prefix + "_self_attention");
    lgraph = connectLayers(lgraph, previous, prefix + "_add_attention/in1");
    lgraph = connectLayers(lgraph, prefix + "_self_attention", prefix + "_add_attention/in2");
    lgraph = connectLayers(lgraph, prefix + "_add_attention", prefix + "_ln_attention");
    lgraph = connectLayers(lgraph, prefix + "_ln_attention", prefix + "_ffn_expand");
    lgraph = connectLayers(lgraph, prefix + "_ln_attention", prefix + "_add_ffn/in1");
    lgraph = connectLayers(lgraph, prefix + "_ffn_project", prefix + "_add_ffn/in2");
    lgraph = connectLayers(lgraph, prefix + "_add_ffn", prefix + "_ln_ffn");
    previous = prefix + "_ln_ffn";
end
lgraph = connectLayers(lgraph, previous, 'last_step');
if includeRegression
    lgraph = connectLayers(lgraph, 'fc', 'regression');
end
end

function [XCell, Y] = makeOneStepDataset(ctx, windowSize, targetKind, useHybrid)
data = ctx.trainScaled;
availableSamples = size(data, 1) - windowSize;
indices = sampleIndices(availableSamples, ctx.cfg.smokeTrainLimit);
numSamples = numel(indices);
XCell = cell(numSamples, 1);
Y = zeros(numSamples, ctx.cfg.N);
for row = 1:numSamples
    i = indices(row);
    seq = data(i:i+windowSize-1, :);
    trueNext = data(i+windowSize, :);
    XCell{row} = seq.';
    Y(row, :) = trainingTarget(seq, trueNext, ctx, targetKind, useHybrid);
end
end

function target = trainingTarget(seq, trueNext, ctx, targetKind, useHybrid)
last = seq(end, :);
base = predictionBase(last, ctx, useHybrid);
switch targetKind
    case "absolute"
        target = trueNext;
    case "residual"
        target = trueNext - last;
    case "hybridResidual"
        target = trueNext - base;
    otherwise
        error("Unknown target kind: %s", targetKind);
end
end

function indices = sampleIndices(availableSamples, sampleLimit)
if isinf(sampleLimit) || availableSamples <= sampleLimit
    indices = 1:availableSamples;
else
    indices = unique(round(linspace(1, availableSamples, sampleLimit)), 'stable');
end
end

function p = teacherForcingProbability(epoch, epochs, scheduleKind)
switch scheduleKind
    case "linear"
        p = max(0.0, 1.0 - (epoch / (epochs * 0.8)));
    case "refined"
        k = 10.0;
        p = k / (k + exp(epoch / (epochs / 10)));
    otherwise
        p = 1.0;
end
end

function [rollout, oneStep] = evaluateSequenceModel(model, ctx, windowSize, modelName)
[testX, testTrue] = makeEvalWindows(ctx.testScaled, windowSize);
preds = predictNextBatch(model, testX);
oneStep = oneStepMetrics(testTrue, preds);
rollout = rolloutSequenceModel(model, ctx, windowSize, modelName);
end

function [XCell, Y] = makeEvalWindows(data, windowSize)
numSamples = size(data, 1) - windowSize;
XCell = cell(numSamples, 1);
Y = zeros(numSamples, size(data, 2));
for i = 1:numSamples
    XCell{i} = data(i:i+windowSize-1, :).';
    Y(i, :) = data(i+windowSize, :);
end
end

function preds = predictNextBatch(model, XCell)
preds = zeros(numel(XCell), model.cfg.N);
for i = 1:numel(XCell)
    seq = XCell{i}.';
    raw = predictRaw(model, seq);
    preds(i, :) = composePrediction(model, seq, raw);
end
end

function raw = predictRaw(model, seq)
if model.isCustom
    X = zeros(model.cfg.N, 1, size(seq, 1), 'single');
    X(:, 1, :) = reshape(single(seq.'), model.cfg.N, 1, size(seq, 1));
    out = predict(model.net, dlarray(X, "CBT"));
    raw = gather(extractdata(out)).';
else
    out = predict(model.net, {seq.'}, 'MiniBatchSize', 1);
    if iscell(out)
        out = out{1};
    end
    raw = double(out);
    if size(raw, 1) > 1
        raw = raw.';
    end
end
raw = double(raw(:)).';
end

function rollout = rolloutSequenceModel(model, ctx, windowSize, modelName)
rolloutSteps = ctx.cfg.rolloutSteps;
currentSeq = ctx.testScaled(1:windowSize, :);
trueScaled = ctx.testScaled(windowSize+1:windowSize+rolloutSteps, :);
predsScaled = zeros(rolloutSteps, ctx.cfg.N);
for step = 1:rolloutSteps
    raw = predictRaw(model, currentSeq);
    predNext = composePrediction(model, currentSeq, raw);
    predsScaled(step, :) = predNext;
    currentSeq = [currentSeq(2:end, :); predNext]; %#ok<AGROW>
end
trueUnscaled = unscaleData(trueScaled, ctx.mu, ctx.sigma);
predUnscaled = unscaleData(predsScaled, ctx.mu, ctx.sigma);
errors = vecnorm(trueUnscaled - predUnscaled, 2, 2);
rollout = table((1:rolloutSteps)', cumulativeRmse(errors), ...
    'VariableNames', {'step','cumulative_rmse'});
if nargin >= 4 && modelName ~= ""
    rollout.Properties.Description = char(modelName);
end
end

function predNext = composePrediction(model, seq, raw)
last = seq(end, :);
switch model.targetKind
    case "absolute"
        predNext = raw;
    case "residual"
        residual = raw;
        predNext = last + residual; % x_last_scaled + residual
    case "hybridResidual"
        residual = raw;
        physics_next_scaled = predictionBase(last, model, true);
        predNext = physics_next_scaled + residual; % physics_next_scaled + residual
    otherwise
        error("Unknown target kind: %s", model.targetKind);
end
end

function predNext = composePredictionDl(raw, currentSeq, ctx, targetKind, useHybrid)
last = squeeze(currentSeq(:, :, end));
switch targetKind
    case "absolute"
        predNext = raw;
    case "residual"
        residual = raw;
        predNext = last + residual; % x_last_scaled + residual
    case "hybridResidual"
        residual = raw;
        physics_next_scaled = physicsNextScaledDl(last, ctx, ctx.cfg.FImperfect);
        predNext = physics_next_scaled + residual; % physics_next_scaled + residual
    otherwise
        error("Unknown target kind: %s", targetKind);
end
end

function X = makeDlSequenceBatch(data, batchIndices, windowSize)
N = size(data, 2);
B = numel(batchIndices);
X = zeros(N, B, windowSize, 'single');
for b = 1:B
    i = batchIndices(b);
    X(:, b, :) = reshape(single(data(i:i+windowSize-1, :).'), N, 1, windowSize);
end
end

function Y = makeDlFutureBatch(data, batchIndices, windowSize, step)
N = size(data, 2);
B = numel(batchIndices);
Y = zeros(N, B, 'single');
for b = 1:B
    i = batchIndices(b);
    Y(:, b) = single(data(i+windowSize+step-1, :).');
end
end

function gradients = clipGradients(gradients, threshold)
total = dlarray(single(0));
for i = 1:size(gradients, 1)
    grad = gradients.Value{i};
    if ~isempty(grad)
        total = total + sum(grad .^ 2, 'all');
    end
end
total = sqrt(total);
if double(gather(extractdata(total))) <= threshold
    return;
end
scale = threshold ./ (total + eps('single'));
for i = 1:size(gradients, 1)
    if ~isempty(gradients.Value{i})
        gradients.Value{i} = gradients.Value{i} .* scale;
    end
end
end

function base = predictionBase(lastScaled, obj, useHybrid)
if useHybrid
    base = physicsNextScaled(lastScaled, obj, obj.cfg.FImperfect);
else
    base = lastScaled;
end
end

function nextScaled = physicsNextScaled(lastScaled, obj, forcing)
lastUnscaled = unscaleData(lastScaled, obj.mu, obj.sigma);
nextUnscaled = rk4Lorenz96Step(lastUnscaled, obj.cfg.dt, forcing);
nextScaled = (nextUnscaled - obj.mu) ./ obj.sigma;
end

function nextScaled = physicsNextScaledDl(lastScaled, obj, forcing)
lastUnscaled = unscaleDataDl(lastScaled, obj.mu, obj.sigma);
nextUnscaled = rk4Lorenz96StepDl(lastUnscaled, obj.cfg.dt, forcing);
mu = single(obj.mu(:));
sigma = single(obj.sigma(:));
nextScaled = (nextUnscaled - mu) ./ sigma;
end

function next = rk4Lorenz96StepDl(x, dt, forcing)
k1 = lorenz96RhsBatchDl(x, forcing);
k2 = lorenz96RhsBatchDl(x + 0.5 * dt * k1, forcing);
k3 = lorenz96RhsBatchDl(x + 0.5 * dt * k2, forcing);
k4 = lorenz96RhsBatchDl(x + dt * k3, forcing);
next = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
end

function rhs = lorenz96RhsBatchDl(x, forcing)
xp1 = x([2:end, 1], :);
xm1 = x([end, 1:end-1], :);
xm2 = x([end-1:end, 1:end-2], :);
rhs = (xp1 - xm2) .* xm1 - x + forcing;
end

function data = unscaleData(dataScaled, mu, sigma)
data = dataScaled .* sigma + mu;
end

function data = unscaleDataDl(dataScaled, mu, sigma)
mu = single(mu(:));
sigma = single(sigma(:));
data = dataScaled .* sigma + mu;
end

function metrics = oneStepMetrics(yTrue, yPred)
diff = yTrue - yPred;
rmse = sqrt(mean(diff.^2, 'all'));
ssRes = sum(diff.^2, 'all');
ssTot = sum((yTrue - mean(yTrue, 'all')).^2, 'all');
r2 = 1 - ssRes / ssTot;
metrics = struct('rmse', rmse, 'r2', r2);
end

function row = phaseRow(label, oneStep, finalRmse)
row = table(label, oneStep.rmse, oneStep.r2, finalRmse, ...
    'VariableNames', {'model_stage','one_step_rmse','one_step_r2','final_rollout_rmse'});
end

function rolloutWithModel = withModelColumn(rollout, modelName)
rolloutWithModel = addvars(rollout, repmat(modelName, height(rollout), 1), 'Before', 1, 'NewVariableNames', 'model');
end

function model = modelStruct(net, isCustom, arch, targetKind, useHybrid, usePinn, useScheduled, windowSize, ctx)
model = struct('net', net, 'isCustom', isCustom, 'arch', arch, 'targetKind', targetKind, ...
    'useHybrid', useHybrid, 'usePinn', usePinn, 'useScheduled', useScheduled, ...
    'windowSize', windowSize, 'mu', ctx.mu, 'sigma', ctx.sigma, 'cfg', ctx.cfg);
end

function plotLorenz96Figures(resultsDir, advancedDir, figuresDir)
summary = readtable(fullfile(resultsDir, "lstm_phase3_summary.csv"));
fig = figure('Visible','off');
bar(summary.window_size, summary.final_rollout_rmse, 0.55);
grid on; xlabel('Window size w'); ylabel('500-step final cumulative RMSE');
title('Lorenz-96 窗口长度消融（MATLAB 训练结果）');
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
title('Lorenz-96 高级模型 500-step rollout 对比（MATLAB 训练结果）');
exportgraphics(fig, fullfile(figuresDir, "lorenz96_advanced_rollout_comparison.png"), 'Resolution', 300);
close(fig);
end

function writeRunStatus(resultsDir)
rows = [
    "all_lorenz96_models", "MATLAB training", "Lorenz-96 CSV files are generated by runLorenz96Experiments.m."
];
runStatus = array2table(rows, 'VariableNames', {'artifact','provenance','note'});
writetable(runStatus, fullfile(resultsDir, "lorenz96_matlab_run_status.csv"));
end
