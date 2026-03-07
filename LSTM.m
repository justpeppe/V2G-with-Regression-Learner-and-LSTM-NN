%% ========================================================================
% LSTM Neural Network for V2G Energy Consumption Prediction
% ========================================================================
%
% WHAT IS THIS SCRIPT?
% This script builds, trains, and evaluates a Long Short-Term Memory (LSTM)
% neural network to predict the energy consumption of a specific zone
% (e.g., Zone 9 - Trieste) in the context of Vehicle-to-Grid (V2G) systems.
%
% WHAT IS AN LSTM?
% LSTM is an advanced Recurrent Neural Network (RNN) architecture designed
% to learn long-term dependencies in sequential data. Unlike standard models,
% LSTMs have internal "memory cells" that maintain state over time, making
% them highly effective for time-series forecasting where past temporal
% patterns strongly influence future outcomes.
%
% WHAT DATA DOES IT USE?
% The network is trained on multivariate time-series data at half-hourly
% (30-minute) intervals. The primary predictors (features) depend on the
% 'useAutoregressive' toggle and can include:
%   1. Autoregressive Data: AAC_energy (historical energy consumption) — OPTIONAL.
%   2. Weather Data: precipprob (precipitation), temp (temperature), windspeed.
%   3. Calendar Data: holiday_indicator (identifies non-working days).
%   4. Cyclical Time: hour_sin, hour_cos, day_sin, day_cos (mathematical
%      transformations that help the LSTM understand daily and weekly cycles).
%
% NAMING CONVENTION
% Models are saved using standard-compliant lowerCamelCase naming strings
% (<= 32 characters) containing the type and 'HHMM' time (e.g., lstmAutoReg1105).
%
% HOW DOES IT WORK? (PIPELINE)
% 1. Data Loading: Loads zone energy data and merges it with weather/holidays.
% 2. Preprocessing: Normalizes the features (Z-score) and adds cyclical time.
% 3. Splitting: Distributes data into Training, Validation, and Test sets using
%    representative dynamic clustering (to avoid data leakage).
% 4. Sequence Creation: Structures the data into sliding 3D windows (e.g.,
%    48 lags = 24 hours of context) to predict the very next time step.
% 5. Training: Configures architecture (BiLSTM layers, Dropout) and trains
%    the model using MATLAB's `trainnet` function and `adam` optimizer.
% 6. Evaluation: De-normalizes predictions back to kWh, calculates standard
%    metrics (RMSE, MAE, MAPE, R-Squared), and displays visual comparisons.
% 7. Residual Analysis: Calculates the Autocorrelation Function (ACF) to
%    check if the errors are purely random (white noise).
% ========================================================================

%% Clean
clear
%% Data Loading

root = fileparts(mfilename('fullpath'));
addpath(root + "\Scripts");

%zoneId=8; % Zona 1016
zoneId=9; %Zona 214 Trieste
%zoneId=10; % Zona 2004
%zoneId=11; % Zona 2002
%zoneId=12; %Torvergata

datas = loadZoneData(root, zoneId);

%% Constants

numLags = 48; % Retained only for potential config reference, no longer dictates sliding windows.
useBayesianOptimization = false; % Set to true to automatically search for optimal hyperparameters
useAutoregressive = true; % Set to true to include AAC_energy in predictors, false for exogenous only

% Predictors: weather, holidays, time (sine/cosine for cyclicity)
% optionally AAC_energy (consumption history) based on the flag above.
exogenousPredictors = ["precipprob", "temp", "windspeed", "holiday_indicator", ...
    "hour_sin", "hour_cos", "day_sin", "day_cos"];

if useAutoregressive
    predictors = ["AAC_energy", exogenousPredictors];
else
    predictors = exogenousPredictors;
end

target = "AAC_energy";
columnsToNormalize = unique([predictors, target], "stable");

%% Automated Report Setup
currentTimeOrario = char(string(datetime("now", "Format", "HHmm")));
if useAutoregressive
    netName = "lstmAutoReg" + currentTimeOrario;
else
    netName = "lstmExog" + currentTimeOrario;
end

currentDate = char(string(datetime("now", "Format", "yyyy_MM_dd")));
reportDir = fullfile(root, "Sessioni", currentDate, "Report_" + netName);
if ~exist(reportDir, "dir")
    mkdir(reportDir);
end
if ~exist(fullfile(reportDir, "fig"), "dir")
    mkdir(fullfile(reportDir, "fig"));
end
diary(fullfile(reportDir, "Analysis_Log.txt"));

% Automatically select representative validation and test day-pairs.
% The function identifies non-consecutive data clusters and extracts typical
% normal days and Sundays to ensure fair and balanced dataset splits.
[validationDays, testDays] = selectRepresentativeDays(datas);


%% Data Preparation

% Extract explicit temporal features (sine and cosine for cyclicity)
datas.hour_sin = sin(2 * pi * hour(datas.time_vector) / 24);
datas.hour_cos = cos(2 * pi * hour(datas.time_vector) / 24);
datas.day_sin  = sin(2 * pi * day(datas.time_vector, 'dayofweek') / 7);
datas.day_cos  = cos(2 * pi * day(datas.time_vector, 'dayofweek') / 7);

[training, validation, test] = splitTrainValTest(datas, validationDays, testDays);

[trainingNorm, testNorm, normParams] = normalizeZScore(training, test, columnsToNormalize);

[~, validationNorm, ~] = normalizeZScore(training, validation, columnsToNormalize);

[xTrain,      yTrain,      timeVectorTrain]       = createLstmSequences(trainingNorm,    predictors, target);
[xTest,       yTest,       timeVectorTest]        = createLstmSequences(testNorm,        predictors, target);
[xValidation, yValidation, timeVectorValidation]  = createLstmSequences(validationNorm,  predictors, target);

%% Sequence Diagnostics

fprintf("\n--- Sequence Diagnostics ---\n");
fprintf("xTrain:      %4d sequences | shape: [%d x %d]\n", numel(xTrain),      size(xTrain{1},1),      size(xTrain{1},2));
if ~isempty(xValidation)
    fprintf("xValidation: %4d sequences | shape: [%d x %d]\n", numel(xValidation),  size(xValidation{1},1), size(xValidation{1},2));
end
if ~isempty(xTest)
    fprintf("xTest:       %4d sequences | shape: [%d x %d]\n", numel(xTest),        size(xTest{1},1),       size(xTest{1},2));
end
if isempty(xValidation)
    error("LSTM:emptyValidation", "xValidation is empty after filtering — check validationDays or context buffer.");
end
if isempty(xTest)
    error("LSTM:emptyTest", "xTest is empty after filtering — check testDays or context buffer.");
end
fprintf("----------------------------\n\n");


%% LSTM Network Definition and Training

numFeatures = size(xTrain{1}, 2);
fprintf("numFeatures:   %d\n", numFeatures);
numResponses = size(yTrain{1}, 2);

if useBayesianOptimization
    fprintf("\n=== Starting Bayesian Optimization ===\n");
    fprintf("This process will test multiple configurations to find the optimal hyperparameters.\n");

    % Define the hyperparameter search space
    optimVars = [
        optimizableVariable('numHiddenUnits',   [50, 300],  'Type', 'integer')
        optimizableVariable('numLayers',        [1, 3],     'Type', 'integer')
        optimizableVariable('initialLearnRate', [1e-4, 1e-2], 'Transform', 'log')
        % --- Aggiunte consigliate ---
        optimizableVariable('dropoutRate',      [0.0, 0.5])
        optimizableVariable('miniBatchSize',    [16, 128],  'Type', 'integer')
        ];


    % The objective function needs access to the training and validation data
    % Since objective functions in bayesopt must accept a single argument (the hyperparameters),
    % we use an anonymous function to pass the data along.
    objFcn = @(params) lstmObjectiveFunction(params, xTrain, yTrain, xValidation, yValidation, numFeatures, numResponses);

    % Run optimization
    bayesResults = bayesopt(objFcn, optimVars, ...
        'MaxObjectiveEvaluations', 100, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'UseParallel', false);


    % Extract best parameters
    bestParams = bayesResults.XAtMinObjective;
    numHiddenUnits = bestParams.numHiddenUnits;
    numLayers = bestParams.numLayers;
    initialLearnRate = bestParams.initialLearnRate;
    drop = bestParams.dropoutRate;
    mb = bestParams.miniBatchSize;
    valFreq = max(1, floor(numel(xTrain) / mb)); % ~1 validation per epoch

    fprintf("\n=== Bayesian Optimization Complete ===\n");
    fprintf("Top 3 Configurations Evaluated:\n");

    resultsTable = bayesResults.XTrace;
    resultsTable.Objective = bayesResults.ObjectiveTrace;
    sortedResults = sortrows(resultsTable, 'Objective');
    head(sortedResults, 3)

    fprintf("\nOptimal Parameters Found:\n");
    fprintf("  Hidden Units      : %d\n", numHiddenUnits);
    fprintf("  LSTM Layers       : %d\n", numLayers);
    fprintf("  Initial Learn Rate: %.5f\n", initialLearnRate);
    fprintf("  Dropout Rate      : %.2f\n", drop);
    fprintf("  Mini Batch Size   : %d\n", mb);
    fprintf("  Validation Freq   : %d\n", valFreq);



else
    fprintf("\nUsing default manual network hyperparameters.\n");
    % Default values from Bayesian Optimization (100 trials, Objective = 0.30188)
    numHiddenUnits = 119;
    numLayers = 1;
    initialLearnRate = 0.01;
    drop = 0.33;
    mb = 30;
    valFreq = max(1, floor(numel(xTrain) / mb)); % ~1 validation per epoch
end

% --- LSTM Network Architecture (MathWorks Seq2Seq pattern) ---
% - numLayers LSTM layers, each with constant numHiddenUnits
% - bottleneck FC(100) + ReLU + Dropout(0.5) to compress representations
% - output FC(numResponses) for the final sequence prediction
layers = [sequenceInputLayer(numFeatures, Normalization="none")];

for i = 1:numLayers
    layers = [layers
        lstmLayer(numHiddenUnits, OutputMode="sequence")];
end

layers = [layers
    fullyConnectedLayer(100)
    reluLayer()
    dropoutLayer(drop)
    fullyConnectedLayer(numResponses)];

% --- Training Options ---
options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    MiniBatchSize=mb, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    InitialLearnRate=initialLearnRate, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=50, ...
    GradientThreshold=1, ...
    ValidationData={xValidation, yValidation}, ...
    ValidationFrequency=valFreq, ...
    ValidationPatience=20, ...
    Metrics=["rsquared", "rmse", "mape", "mae", "mse"], ...
    Plots="training-progress", ...
    Verbose=false ...
    );

[net, info] = trainnet(xTrain, yTrain, layers, "mse", options);

% Intercept the Training Progress figure (which has a dynamic Name with timestamps)
allFigs = findall(groot, 'Type', 'Figure');
for i = 1:numel(allFigs)
    if contains(allFigs(i).Name, "Training Progress", "IgnoreCase", true)
        exportgraphics(allFigs(i), fullfile(reportDir, "Training_Progress.png"), Resolution=300);
        savefig(allFigs(i), fullfile(reportDir, "fig", "Training_Progress.fig"));
        break;
    end
end

%% Model Saving
% The struct contains everything needed to reproduce and evaluate the network:
%   - net:         the trained network (use for predictions with minibatchpredict)
%   - indicators:  training and validation metrics (RMSE, R^2, MAPE, MAE, MSE)
%   - normParams:  z-score parameters to de-normalize predictions
%   - x/yTrain|Validation|Test: sequences and targets ready for future predictions
%   - timeVector*: timestamps corresponding to each sequence (for plotting)
%   - config:      network configuration parameters (hidden units, features, etc.)

%   - config:      network configuration parameters (hidden units, features, etc.)

indicators = getBestIndicators(info);

netStruct = struct( ...
    "zoneId",             zoneId, ...
    "net",                net, ...
    "layers",             {layers}, ...
    "options",            options, ...
    "indicators",         indicators, ...
    "normParams",         normParams, ...
    "xTrain",             {xTrain}, ...
    "yTrain",             {yTrain}, ...
    "timeVectorTrain",    {timeVectorTrain}, ...
    "xValidation",        {xValidation}, ...
    "yValidation",        {yValidation}, ...
    "timeVectorValidation", {timeVectorValidation}, ...
    "xTest",              {xTest}, ...
    "yTest",              {yTest}, ...
    "timeVectorTest",     {timeVectorTest}, ...
    "config",             struct( ...
    "numFeatures",    numFeatures, ...
    "numResponses",   numResponses, ...
    "numHiddenUnits", numHiddenUnits, ...
    "numLayers",      numLayers, ...
    "numLags",        numLags, ...
    "predictors",     predictors, ...
    "target",         target ...
    ) ...
    );

models = struct();
models.(netName) = netStruct;

%% Prediction and Result Visualization
% Generate predictions on training, validation, and test datasets.
% The plotResults function automatically denormalizes the data for plotting.

m = models.(netName); % Shorthand alias for the current model

% Generates cell arrays for Seq2Seq, we concatenate them to use in plotting and metrics
trainingPrediction   = minibatchpredict(m.net, m.xTrain, UniformOutput=false, SequencePaddingDirection="left");
validationPrediction = minibatchpredict(m.net, m.xValidation, UniformOutput=false, SequencePaddingDirection="left");
testPrediction       = minibatchpredict(m.net, m.xTest, UniformOutput=false, SequencePaddingDirection="left");

% Remove padding from the predictions before unrolling
for n = 1:numel(trainingPrediction)
    seqLen = size(m.yTrain{n}, 1);
    trainingPrediction{n} = trainingPrediction{n}(end-seqLen+1:end, :);
end
for n = 1:numel(validationPrediction)
    seqLen = size(m.yValidation{n}, 1);
    validationPrediction{n} = validationPrediction{n}(end-seqLen+1:end, :);
end
for n = 1:numel(testPrediction)
    seqLen = size(m.yTest{n}, 1);
    testPrediction{n} = testPrediction{n}(end-seqLen+1:end, :);
end

% Unroll cells into long matrices for plotting and analysis
trainingPrediction = cell2mat(trainingPrediction);
validationPrediction = cell2mat(validationPrediction);
testPrediction = cell2mat(testPrediction);

yTrainMat = cell2mat(m.yTrain);
yValidationMat = cell2mat(m.yValidation);
yTestMat = cell2mat(m.yTest);

timeVectorTrainMat = vertcat(m.timeVectorTrain{:});
timeVectorValidationMat = vertcat(m.timeVectorValidation{:});
timeVectorTestMat = vertcat(m.timeVectorTest{:});

trainingFig   = plotResults(trainingPrediction,   yTrainMat,      m.normParams, target, timeVectorTrainMat);
validationFig = plotResults(validationPrediction, yValidationMat, m.normParams, target, timeVectorValidationMat);
testFig       = plotResults(testPrediction,        yTestMat,       m.normParams, target, timeVectorTestMat);

exportgraphics(trainingFig, fullfile(reportDir, "Seq_01_Training.png"), Resolution=300);
exportgraphics(validationFig, fullfile(reportDir, "Seq_02_Validation.png"), Resolution=300);
exportgraphics(testFig, fullfile(reportDir, "Seq_03_Test.png"), Resolution=300);

savefig(trainingFig, fullfile(reportDir, "fig", "Seq_01_Training.fig"));
savefig(validationFig, fullfile(reportDir, "fig", "Seq_02_Validation.fig"));
savefig(testFig, fullfile(reportDir, "fig", "Seq_03_Test.fig"));

%% Test Set Metrics Calculation
% Compute RMSE, R^2, MAPE, MAE, and MSE on the denormalized test predictions (in actual kWh).
models.(netName).indicators.Test = computeMetrics(testPrediction, yTestMat, normParams, char(target));

%% Final Model Save
% Save the complete model structure including the newly calculated test metrics.
saveModels(root, models, char(netName), []);

%% Final Summary in Command Window
ind = models.(netName).indicators;
cfg = models.(netName).config;

fprintf("\n========================================================\n");
fprintf("          LSTM MODEL SUMMARY — %s\n", netName);
fprintf("--------------------------------------------------------\n");
fprintf("   Zone:               %d\n",      zoneId);
fprintf("   Features (%d):      %s\n",      cfg.numFeatures, strjoin(cfg.predictors, ", "));
fprintf("   Target:             %s\n",      cfg.target);
fprintf("   Lags (window):      %d samples\n", cfg.numLags);
fprintf("   Layers:             %d LSTM layer(s)\n", cfg.numLayers);
layerLabel = "";
for li = 1:cfg.numLayers
    layerLabel = layerLabel + sprintf("L%d: %d units  ", li, cfg.numHiddenUnits);
end
fprintf("   Hidden units:       %s\n", strtrim(layerLabel));
fprintf("   Head:               FC(100) → ReLU → Dropout(%.2f) → FC(%d)\n", drop, cfg.numResponses);
fprintf("--------------------------------------------------------\n");
fprintf("   TRAINING METRICS (normalized, z-score)\n");
fprintf("     R²:    %.4f\n",    ind.Training.Rsquared);
fprintf("     RMSE:  %.4f\n",    ind.Training.RMSE);
fprintf("     MAE:   %.4f\n",    ind.Training.MAE);
fprintf("     MAPE:  %.2f %%\n", ind.Training.MAPE);
fprintf("--------------------------------------------------------\n");
fprintf("   VALIDATION METRICS (normalized, z-score)\n");
fprintf("     R²:    %.4f\n",    ind.Validation.Rsquared);
fprintf("     RMSE:  %.4f\n",    ind.Validation.RMSE);
fprintf("     MAE:   %.4f\n",    ind.Validation.MAE);
fprintf("     MAPE:  %.2f %%\n", ind.Validation.MAPE);
fprintf("--------------------------------------------------------\n");
fprintf("   TEST METRICS (denormalized, actual kWh)\n");
fprintf("     R²:    %.4f\n",      ind.Test.RSquared);
fprintf("     RMSE:  %.4f kWh\n",  ind.Test.RMSE);
fprintf("     MAE:   %.4f kWh\n",  ind.Test.MAE);
fprintf("     MAPE:  %.2f %%\n",   ind.Test.MAPE);
fprintf("========================================================\n\n");

%% Analysis Section 1: Target Autocorrelation
% The autocorrelation of AAC_energy justifies the choice of numLags=48:
% a significant coefficient at lag 48 indicates that the consumption
% 24 hours ago is highly predictive of the current value.

targetMu  = normParams.(char(target)).mean;
targetStd = normParams.(char(target)).std;

% Denormalize targets (convert back to real kWh scale)
yTestReal       = yTestMat       .* targetStd + targetMu;
yValidationReal = yValidationMat .* targetStd + targetMu;
yTrainReal      = yTrainMat      .* targetStd + targetMu;

% Denormalize predictions
testPredReal       = testPrediction       .* targetStd + targetMu;
validationPredReal = validationPrediction .* targetStd + targetMu;

figACF_Target = figure("Name", "Target Autocorrelation", "NumberTitle", "off");
autocorr(yTrainReal, NumLags=100);
title("Autocorrelation of AAC\_energy (Training Set)", FontSize=13);
xlabel("Lag (30-min time steps)");
ylabel("Autocorrelation Coefficient");
xline(48, "--r", "Lag 48 (24h)", LabelVerticalAlignment="bottom");
grid on;
exportgraphics(figACF_Target, fullfile(reportDir, "Analysis_01_Target_ACF.png"), Resolution=300);
savefig(figACF_Target, fullfile(reportDir, "fig", "Analysis_01_Target_ACF.fig"));

%% Analysis Section 2: Residual Analysis on Test Set
% Residuals should ideally resemble "white noise" with no systematic patterns.
% Structured behavior in residuals indicates the network missed some underlying info.

residuiTest = double(yTestReal - testPredReal);

figResid = figure("Name", "Residual Analysis — Test Set", "NumberTitle", "off", ...
    "Position", [100 100 1100 600]);
tiledlayout(2, 2, TileSpacing="compact", Padding="compact");

nexttile;
% Using scatter instead of plot: since test days are not contiguous,
% plot() would connect distant dates with straight lines creating visual artifacts.
scatter(timeVectorTestMat, residuiTest, 20, [0.2 0.5 0.9], "filled", MarkerFaceAlpha=0.75);
yline(0, "--k", LineWidth=1);
title("Residuals Over Time (scatter)", FontSize=12);
xlabel("Date"); ylabel("Error (kWh)");
grid on;

nexttile;
histogram(residuiTest, 30, FaceColor=[0.2 0.7 0.5], EdgeColor="none");
title("Residual Distribution", FontSize=12);
xlabel("Error (kWh)"); ylabel("Frequency");
grid on;

% Residual ACF: autocorr() ignores nexttile if called without output arguments.
% We calculate the numerical values and plot them manually instead.
numLagsAcf = min(50, floor(numel(residuiTest) / 2) - 1);
[acfVals, acfLags, acfBounds] = autocorr(residuiTest, NumLags=numLagsAcf);
axAcf = nexttile;
stem(axAcf, acfLags, acfVals, "filled", MarkerSize=4, Color=[0.2 0.5 0.9]);
yline(axAcf,  acfBounds(1), "--", Color=[0.85 0.5 0.1], LineWidth=1);
yline(axAcf, -acfBounds(1), "--", Color=[0.85 0.5 0.1], LineWidth=1);
yline(axAcf, 0, "-k", LineWidth=0.6);
title(axAcf, "Residuals Autocorrelation", FontSize=12);
xlabel(axAcf, "Lag"); ylabel(axAcf, "ACF");
grid(axAcf, "on");

% Q-Q Plot: explicitly pass the axis to respect the nexttile layout
axQQ = nexttile;
qqplot(axQQ, residuiTest);
title(axQQ, "Q-Q Plot for Residuals vs Normal", FontSize=12);
grid(axQQ, "on");

sgtitle(sprintf("Residual Analysis — Test Set (RMSE=%.2f kWh)", ind.Test.RMSE), ...
    FontSize=14, FontWeight="bold");
exportgraphics(figResid, fullfile(reportDir, "Analysis_02_Residuals.png"), Resolution=300);
savefig(figResid, fullfile(reportDir, "fig", "Analysis_02_Residuals.fig"));

%% Analysis Section 3: Scatter Plot Real vs Predicted (Test Set)
% Points should ideally align along the bisector y=x.
% The dispersion around this line represents the magnitude of the average error.

figScatter = figure("Name", "Scatter Plot: Real vs Predicted — Test Set", "NumberTitle", "off", ...
    "Position", [200 200 600 580]);

scatter(yTestReal, testPredReal, 35, [0.15 0.45 0.85], "filled", ...
    MarkerFaceAlpha=0.7);
hold on;
lims = [min([yTestReal; testPredReal]) * 0.95, max([yTestReal; testPredReal]) * 1.05];
plot(lims, lims, "--r", LineWidth=1.8, DisplayName="y = x Bisector");
hold off;

xlabel("Real Consumption (kWh)", FontSize=12);
ylabel("Predicted Consumption (kWh)", FontSize=12);
title(sprintf("Test Set — R² = %.4f  |  RMSE = %.2f kWh", ...
    ind.Test.RSquared, ind.Test.RMSE), FontSize=13);
legend("Predictions", "y = x", Location="northwest");
axis equal; axis tight; grid on;
exportgraphics(figScatter, fullfile(reportDir, "Analysis_03_Scatter.png"), Resolution=300);
savefig(figScatter, fullfile(reportDir, "fig", "Analysis_03_Scatter.fig"));

%% Analysis Section 4: Error Breakdown by Hour
% Identifies during which hours of the day the network makes the largest errors.
% Error peaks at sunrise/sunset often indicate transitions that are hard to model.

orarioTest    = hour(timeVectorTestMat);
erroreAssoluto = abs(residuiTest);

errorePerOra = zeros(24, 1);
stdPerOra    = zeros(24, 1);
for h = 0:23
    maschera = (orarioTest == h);
    if any(maschera)
        errorePerOra(h + 1) = mean(erroreAssoluto(maschera));
        stdPerOra(h + 1)    = std(erroreAssoluto(maschera));
    end
end

figHourly = figure("Name", "Hourly Error Breakdown", "NumberTitle", "off", ...
    "Position", [300 200 900 420]);

ore = 0:23;

bar(ore, errorePerOra, FaceColor=[0.95 0.6 0.2], EdgeColor="none", BarWidth=0.7);
hold on;
errorbar(ore, errorePerOra, stdPerOra / 2, stdPerOra / 2, ...
    "k.", LineWidth=1.2, CapSize=4);
hold off;

xlabel("Hour of Day (0–23)", FontSize=12);
ylabel("Mean MAE (kWh)", FontSize=12);
title("Hourly Error Breakdown — Test Set", FontSize=13);
xticks(0:23);
xticklabels(string(0:23) + ":00");
xtickangle(45);
grid on; grid minor;
exportgraphics(figHourly, fullfile(reportDir, "Analysis_04_HourlyError.png"), Resolution=300);
savefig(figHourly, fullfile(reportDir, "fig", "Analysis_04_HourlyError.fig"));

% Highlight the hour with the highest error
[~, oraMax] = max(errorePerOra);
fprintf("\nHour with maximum average error: %02d:00 (MAE = %.2f kWh)\n", ...
    oraMax - 1, errorePerOra(oraMax));
fprintf("Hour with minimum average error: %02d:00 (MAE = %.2f kWh)\n\n", ...
    find(errorePerOra == min(errorePerOra), 1) - 1, min(errorePerOra));

%% Outlier Analysis — Investigation of Samples with Highest Error
% Identifies the timestamps where the network made the largest errors and
% displays the corresponding raw data (weather + real consumption) to
% investigate the root cause.

nTopOutliers = 10;  % Number of samples to investigate
erroreAssolutoTest = abs(residuiTest);
[errOrd, idxOrd] = sort(erroreAssolutoTest, "descend");
idxWorst = idxOrd(1:min(nTopOutliers, numel(idxOrd)));

fprintf("\n=== Top %d Outliers — Highest Absolute Error Samples ===\n", nTopOutliers);
fprintf("%-22s  %10s  %10s  %10s  %8s  %8s  %10s\n", ...
    "Timestamp", "Real(kWh)", "Pred(kWh)", "Err(kWh)", "Temp(C)", "Wind", "PrecipPr");

% Find corresponding rows in the raw data for each timestamp
for k = 1:numel(idxWorst)
    ts  = timeVectorTestMat(idxWorst(k));
    righe = (datas.time_vector == ts);
    if any(righe)
        r = find(righe, 1);
        fprintf("%-22s  %10.2f  %10.2f  %10.2f  %8.1f  %8.1f  %10.1f\n", ...
            string(ts, "yyyy-MM-dd HH:mm"), ...
            double(yTestReal(idxWorst(k))), ...
            double(testPredReal(idxWorst(k))), ...
            double(residuiTest(idxWorst(k))), ...
            datas.temp(r), datas.windspeed(r), datas.precipprob(r));
    end
end
fprintf("%s\n\n", repmat('-', 1, 86));

%% Persistence Baseline — y(t) = y(t-1)
% Verifies if the LSTM actually learns logic beyond a simple persistence rule.

yPers        = double(yTestReal(1:end-1));
yTestShifted = double(yTestReal(2:end));

ssResPers = sum((yTestShifted - yPers).^2);
ssTotPers = sum((yTestShifted - mean(yTestShifted)).^2);
r2Pers    = 1 - ssResPers / ssTotPers;
rmsePers  = sqrt(mean((yTestShifted - yPers).^2));
maePers   = mean(abs(yTestShifted - yPers));

fprintf("\n--- PERSISTENCE BASELINE COMPARISON ---\n");
fprintf("%-15s  %8s  %12s  %12s\n", "Model", "R^2", "RMSE (kWh)", "MAE (kWh)");
fprintf("%s\n", repmat('-', 1, 52));
fprintf("%-15s  %8.4f  %12.2f  %12.2f\n", "LSTM", ind.Test.RSquared, ind.Test.RMSE, ind.Test.MAE);
fprintf("%-15s  %8.4f  %12.2f  %12.2f\n", "Persistence", r2Pers, rmsePers, maePers);
fprintf("%s\n", repmat('-', 1, 52));

diary off;

%% Local Functions

function valError = lstmObjectiveFunction(params, xTrain, yTrain, xValidation, yValidation, numFeatures, numResponses)
% Evaluate the network with the given hyperparameters.
% The objective is to minimize the Validation RMSE.

% Define architecture based on bayesian parameters
numHiddenUnits = params.numHiddenUnits;
numLayers = params.numLayers;

layers = [sequenceInputLayer(numFeatures, Normalization="none")];
for i = 1:numLayers
    layers = [layers
        lstmLayer(numHiddenUnits, OutputMode="sequence")];
end

layers = [layers
    fullyConnectedLayer(100)
    reluLayer()
    dropoutLayer(params.dropoutRate)
    fullyConnectedLayer(numResponses)];

% Define short training options for the objective evaluation
options = trainingOptions("adam", ...
    MaxEpochs=50, ... % Shortened for faster evaluation during search
    MiniBatchSize=params.miniBatchSize, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    InitialLearnRate=params.initialLearnRate, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=25, ...
    GradientThreshold=1, ...
    ValidationData={xValidation, yValidation}, ...
    ValidationFrequency=max(1, floor(numel(xTrain) / params.miniBatchSize)), ...
    ValidationPatience=10, ...
    Verbose=false ...
    );

% Train the network
[net, ~] = trainnet(xTrain, yTrain, layers, "mse", options);

% Calculate validation RMSE
valPredCell = minibatchpredict(net, xValidation, UniformOutput=false, SequencePaddingDirection="left");

% Unpad and compute RMSE
errSqSum = 0;
totalPoints = 0;
for n = 1:numel(valPredCell)
    seqLen = size(yValidation{n}, 1);
    predClean = valPredCell{n}(end-seqLen+1:end, :);
    truth = yValidation{n};
    errSqSum = errSqSum + sum((truth - predClean).^2, 'all');
    totalPoints = totalPoints + numel(truth);
end

valError = sqrt(errSqSum / totalPoints);
end