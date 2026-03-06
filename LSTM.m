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
% (30-minute) intervals. The primary predictors (features) are:
%   1. AAC_energy: Historical energy consumption (autoregressive component).
%   2. Weather Data: precipprob (precipitation), temp (temperature), windspeed.
%   3. Calendar Data: holiday_indicator (identifies non-working days).
%   4. Cyclical Time: hour_sin, hour_cos, day_sin, day_cos (mathematical
%      transformations that help the LSTM understand daily and weekly cycles).
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

zoneId=8; % Zona 1016
%zoneId=9; %Zona 214 Trieste
%zoneId=10; % Zona 2004
%zoneId=11; % Zona 2002
%zoneId=12; %Torvergata

datas = loadZoneData(root, zoneId);

%% Constants

numLags = 48;

% Predictors: weather, holidays, time (sine/cosine for cyclicity)
% and AAC_energy (consumption history). The network has access to y(t-1)...y(t-48),
% but with 48 steps of weather/time context it is incentivized to learn
% real patterns beyond simple persistence.
predictors = ["AAC_energy", "precipprob", "temp", "windspeed", "holiday_indicator", ...
    "hour_sin", "hour_cos", "day_sin", "day_cos"];
target = "AAC_energy";
columnsToNormalize = unique([predictors, target], "stable");

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

% Prepend the last 'numLags' rows from the training set to validation and test sets.
% This provides the necessary historical context so the LSTM can make predictions
% starting from the very first time step of the validation and test datasets.
trainContextRows = trainingNorm(end - numLags + 1:end, :);

[~, validationNorm, ~] = normalizeZScore(training, validation, columnsToNormalize);
validationWithCtx = [trainContextRows; validationNorm];
testWithCtx       = [trainContextRows; testNorm];

[xTrain,      yTrain,      timeVectorTrain]      = createLstmSequences(trainingNorm,    numLags, predictors, target);
[xTest,       yTest,       timeVectorTest]        = createLstmSequences(testWithCtx,      numLags, predictors, target);
[xValidation, yValidation, timeVectorValidation]  = createLstmSequences(validationWithCtx, numLags, predictors, target);

% Keep only samples that belong to the actual validation/test days (discard the context rows)
testDaysNorm = dateshift(testDays, "start", "day");
validationDaysNorm = dateshift(validationDays, "start", "day");
keepTest = ismember(dateshift(timeVectorTest, "start", "day"), testDaysNorm);
keepVal  = ismember(dateshift(timeVectorValidation, "start", "day"), validationDaysNorm);
xTest = xTest(keepTest); yTest = yTest(keepTest, :); timeVectorTest = timeVectorTest(keepTest);
xValidation = xValidation(keepVal); yValidation = yValidation(keepVal, :); timeVectorValidation = timeVectorValidation(keepVal);

%% Sequence Diagnostics

fprintf("\n--- Sequence Diagnostics ---\n");
fprintf("xTrain:      %4d sequences | shape: [%d x %d]\n", numel(xTrain),      size(xTrain{1},1),      size(xTrain{1},2));
fprintf("xValidation: %4d sequences | shape: [%d x %d]\n", numel(xValidation),  size(xValidation{1},1), size(xValidation{1},2));
fprintf("xTest:       %4d sequences | shape: [%d x %d]\n", numel(xTest),        size(xTest{1},1),       size(xTest{1},2));
fprintf("yTrain:      [%d x %d]  |  yValidation: [%d x %d]  |  yTest: [%d x %d]\n", ...
    size(yTrain,1), size(yTrain,2), size(yValidation,1), size(yValidation,2), size(yTest,1), size(yTest,2));
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
numResponses = size(yTrain, 2);

numHiddenUnits = 128;
drop = 0.2;
mb = 64;
valFreq = max(1, ceil(numel(xTrain) / mb));

% --- LSTM Network Architecture ---
% 1. sequenceInputLayer  -> receives sequences [numLags × numFeatures]
% 2. lstmLayer           -> learns temporal dependencies, outputMode="last" returns
%                           only the final state (many-to-one regression)
% 3. dropoutLayer        -> randomly drops 20% of neurons to avoid overfitting
% 4. fullyConnectedLayer -> intermediate dense layer to compose LSTM features
% 5. reluLayer           -> non-linear activation (clips negative values)
% 6. fullyConnectedLayer -> output layer: produces the single predicted value
layers = [
    sequenceInputLayer(numFeatures, Normalization="none")
    lstmLayer(numHiddenUnits, OutputMode="last")
    dropoutLayer(drop)
    fullyConnectedLayer(numHiddenUnits / 2)
    reluLayer
    dropoutLayer(drop)
    fullyConnectedLayer(numResponses)
    ];

% --- Training Options ---
% Adam: adaptive optimizer, excellent for LSTM networks, converges quickly.
% Shuffle="every-epoch": shuffles sequences every epoch so the network does not
%   memorize the seasonal order but generalizes the patterns.
% ValidationPatience=15: stops training if validation loss does not
%   improve for 15 consecutive evaluations (early stopping).
options = trainingOptions("adam", ...
    MaxEpochs=150, ...
    MiniBatchSize=mb, ...
    Shuffle="every-epoch", ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=50, ...
    GradientThreshold=1, ...
    GradientThresholdMethod="l2norm", ...
    L2Regularization=1e-4, ...
    ValidationData={xValidation, yValidation}, ...
    ValidationFrequency=valFreq, ...
    ValidationPatience=15, ...
    Metrics=["rsquared", "rmse", "mape", "mae", "mse"], ...
    Plots="training-progress", ...
    Verbose=false, ...
    ExecutionEnvironment="auto" ...
    );

[net, info] = trainnet(xTrain, yTrain, layers, "mse", options);

%% Model Saving
% The struct contains everything needed to reproduce and evaluate the network:
%   - net:         the trained network (use for predictions with minibatchpredict)
%   - indicators:  training and validation metrics (RMSE, R^2, MAPE, MAE, MSE)
%   - normParams:  z-score parameters to de-normalize predictions
%   - x/yTrain|Validation|Test: sequences and targets ready for future predictions
%   - timeVector*: timestamps corresponding to each sequence (for plotting)
%   - config:      network configuration parameters (hidden units, features, etc.)

ts      = char(string(datetime("now", "Format", "yyyy_MM_dd_HH_mm_ss")));
netName = "net_" + ts;

indicators = getBestIndicators(info);

netStruct = struct( ...
    "zoneId",             zoneId, ...
    "net",                net, ...
    "indicators",         indicators, ...
    "normParams",         normParams, ...
    "xTrain",             {xTrain}, ...
    "yTrain",             yTrain, ...
    "timeVectorTrain",    timeVectorTrain, ...
    "xValidation",        {xValidation}, ...
    "yValidation",        yValidation, ...
    "timeVectorValidation", timeVectorValidation, ...
    "xTest",              {xTest}, ...
    "yTest",              yTest, ...
    "timeVectorTest",     timeVectorTest, ...
    "config",             struct( ...
    "numFeatures",    numFeatures, ...
    "numResponses",   numResponses, ...
    "numHiddenUnits", numHiddenUnits, ...
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

trainingPrediction   = minibatchpredict(m.net, m.xTrain);
validationPrediction = minibatchpredict(m.net, m.xValidation);
testPrediction       = minibatchpredict(m.net, m.xTest);

trainingFig   = plotResults(trainingPrediction,   m.yTrain,      m.normParams, target, m.timeVectorTrain);
validationFig = plotResults(validationPrediction, m.yValidation, m.normParams, target, m.timeVectorValidation);
testFig       = plotResults(testPrediction,        m.yTest,       m.normParams, target, m.timeVectorTest);

%% Test Set Metrics Calculation
% Compute RMSE, R^2, MAPE, MAE, and MSE on the denormalized test predictions (in actual kWh).
models.(netName).indicators.Test = computeMetrics(testPrediction, yTest, normParams, char(target));

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
fprintf("   Hidden units:       %d (L1)  +  %d (L2)\n", cfg.numHiddenUnits, cfg.numHiddenUnits/2);
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
yTestReal       = yTest       .* targetStd + targetMu;
yValidationReal = yValidation .* targetStd + targetMu;
yTrainReal      = yTrain      .* targetStd + targetMu;

% Denormalize predictions
testPredReal       = testPrediction       .* targetStd + targetMu;
validationPredReal = validationPrediction .* targetStd + targetMu;

figure("Name", "Target Autocorrelation", "NumberTitle", "off");
autocorr(yTrainReal, NumLags=100);
title("Autocorrelation of AAC\_energy (Training Set)", FontSize=13);
xlabel("Lag (30-min time steps)");
ylabel("Autocorrelation Coefficient");
xline(48, "--r", "Lag 48 (24h)", LabelVerticalAlignment="bottom");
grid on;

%% Analysis Section 2: Residual Analysis on Test Set
% Residuals should ideally resemble "white noise" with no systematic patterns.
% Structured behavior in residuals indicates the network missed some underlying info.

residuiTest = double(yTestReal - testPredReal);

figure("Name", "Residual Analysis — Test Set", "NumberTitle", "off", ...
    "Position", [100 100 1100 600]);
tiledlayout(2, 2, TileSpacing="compact", Padding="compact");

nexttile;
% Using scatter instead of plot: since test days are not contiguous,
% plot() would connect distant dates with straight lines creating visual artifacts.
scatter(timeVectorTest, residuiTest, 20, [0.2 0.5 0.9], "filled", MarkerFaceAlpha=0.75);
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

%% Analysis Section 3: Scatter Plot Real vs Predicted (Test Set)
% Points should ideally align along the bisector y=x.
% The dispersion around this line represents the magnitude of the average error.

figure("Name", "Scatter Plot: Real vs Predicted — Test Set", "NumberTitle", "off", ...
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

%% Analysis Section 4: Error Breakdown by Hour
% Identifies during which hours of the day the network makes the largest errors.
% Error peaks at sunrise/sunset often indicate transitions that are hard to model.

orarioTest    = hour(timeVectorTest);
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

figure("Name", "Hourly Error Breakdown", "NumberTitle", "off", ...
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
    ts  = timeVectorTest(idxWorst(k));
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

if ind.Test.RSquared > r2Pers
    fprintf("Result: LSTM outperforms persistence baseline.\n\n");
else
    fprintf("Result: LSTM does not outperform persistence baseline.\n\n");
end