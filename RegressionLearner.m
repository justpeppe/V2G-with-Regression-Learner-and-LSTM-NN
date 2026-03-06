%% ========================================================================
% Regression Learner for V2G Energy Consumption Prediction
% ========================================================================
%
% WHAT IS THIS SCRIPT?
% This script builds, trains, and evaluates a classical regression model
% to predict the energy consumption of a specific zone (e.g., Zone 9 -
% Trieste) in the context of Vehicle-to-Grid (V2G) systems.
%
% HOW DOES IT DIFFER FROM LSTM.m?
% While LSTM.m models temporal dependencies as sequences, this script
% flattens the 48-lag context window into a wide feature vector
% (48 lags × 9 features = 432 input columns) and trains a tabular
% regression model. This is the classical "Regression Learner" approach
% used for direct comparison against the LSTM baseline.
%
% WHAT DATA DOES IT USE?
% Identical pipeline to LSTM.m (same zone, same predictors, same split):
%   1. AAC_energy: Historical energy consumption (autoregressive).
%   2. Weather: precipprob, temp, windspeed.
%   3. Calendar: holiday_indicator.
%   4. Cyclical Time: hour_sin, hour_cos, day_sin, day_cos.
%
% HOW DOES IT WORK? (PIPELINE)
% 1. Data Loading: Same as LSTM.m (loadZoneData, selectRepresentativeDays).
% 2. Preprocessing: Z-score normalization, cyclical time features.
% 3. Splitting: Training / Validation / Test via splitTrainValTest.
% 4. Lag Matrix Creation: createRegressionLags produces a wide table
%    with numLags × numFeatures input columns + 1 target column.
% 5. Training: Selectable model type via `modelType` constant.
%    Supported: "ensemble", "svm", "tree", "gpr", "linear".
% 6. Evaluation: Same metrics (RMSE, MAE, MAPE, R²) and plots as LSTM.m,
%    including residual analysis and hourly breakdown.
% 7. Saving: Compatible with saveModels so results can be compared.
% ========================================================================

%% Clean
clear

%% Data Loading

root = fileparts(mfilename("fullpath"));
addpath(root + "\Scripts");

%zoneId=8;  % Zona 1016
zoneId=9;   % Zona 214 Trieste
%zoneId=10; % Zona 2004
%zoneId=11; % Zona 2002
%zoneId=12; % Torvergata

datas = loadZoneData(root, zoneId);

%% Constants

numLags  = 48;   % Context window: 48 × 30-min = 24 hours (matches LSTM.m)

% Model type selector — choose one of:
%   "ensemble"  → Gradient Boosted Trees (fitrensemble, 'LSBoost')
%   "tree"      → Regression Tree       (fitrtree)
%   "svm"       → Support Vector Machine (fitrsvm, RBF kernel)
%   "gpr"       → Gaussian Process Regression (fitrgp)
%   "linear"    → Regularised Linear (fitrlinear, Lasso)
modelType = "ensemble";

% Approach B — Exogenous-only predictors:
% AAC_energy is removed from the feature set entirely. The model must predict
% y(t) purely from weather (precipprob, temp, windspeed, holiday) and
% cyclical time (hour_sin/cos, day_sin/cos). No autoregressive shortcut possible.
exogenousPredictors = ["precipprob", "temp", "windspeed", "holiday_indicator", ...
    "hour_sin", "hour_cos", "day_sin", "day_cos"];
predictors = exogenousPredictors;          % 8 features, no AAC_energy
target     = "AAC_energy";
columnsToNormalize = unique([exogenousPredictors, target], "stable");

% Select representative validation and test days (same function as LSTM.m)
[validationDays, testDays] = selectRepresentativeDays(datas);

%% Data Preparation

% Cyclical time features (identical to LSTM.m)
datas.hour_sin = sin(2 * pi * hour(datas.time_vector) / 24);
datas.hour_cos = cos(2 * pi * hour(datas.time_vector) / 24);
datas.day_sin  = sin(2 * pi * day(datas.time_vector, "dayofweek") / 7);
datas.day_cos  = cos(2 * pi * day(datas.time_vector, "dayofweek") / 7);

% Train / Validation / Test split
[training, validation, test] = splitTrainValTest(datas, validationDays, testDays);

% Z-score normalization fitted on training set only (no data leakage)
[trainingNorm, testNorm,       normParams] = normalizeZScore(training, test,       columnsToNormalize);
[~,            validationNorm, ~         ] = normalizeZScore(training, validation,  columnsToNormalize);

%% Create Lagged Feature Matrices

fprintf("\n--- Creating lagged feature matrices (%d lags × %d features) ---\n", ...
    numLags, numel(predictors));

trainLag = createRegressionLags(trainingNorm,   numLags, predictors, char(target));
valLag   = createRegressionLags(validationNorm, numLags, predictors, char(target));
testLag  = createRegressionLags(testNorm,       numLags, predictors, char(target));

% Separate feature matrix X and target vector y (exclude time_vector column)
featureCols = trainLag.Properties.VariableNames;
featureCols = featureCols(~strcmp(featureCols, "time_vector") & ~strcmp(featureCols, char(target)));

XTrain = table2array(trainLag(:, featureCols));
yTrain = trainLag.(char(target));

XVal   = table2array(valLag(:, featureCols));
yVal   = valLag.(char(target));

XTest  = table2array(testLag(:, featureCols));
yTest  = testLag.(char(target));

timeVectorTest       = testLag.time_vector;
timeVectorVal        = valLag.time_vector;
timeVectorTrainTable = trainLag.time_vector;

numInputFeatures = size(XTrain, 2);
fprintf("numInputFeatures (lags x features): %d\n", numInputFeatures);
fprintf("Training samples:   %d\n", size(XTrain, 1));
fprintf("Validation samples: %d\n", size(XVal, 1));
fprintf("Test samples:       %d\n\n", size(XTest, 1));

% No autoregressive columns to remove: AAC_energy was excluded from predictors.

%% Model Training

fprintf("=== Training model: %s ===\n", modelType);

switch modelType

    case "ensemble"
        % Gradient Boosted Trees — robust to outliers, good baseline
        mdl = fitrensemble(XTrain, yTrain, ...
            "Method",          "LSBoost", ...
            "NumLearningCycles", 300, ...
            "LearnRate",       0.05, ...
            "Learners",        templateTree("MaxNumSplits", 6));

    case "tree"
        % Single regression tree — interpretable, fast
        mdl = fitrtree(XTrain, yTrain, ...
            "MaxNumSplits", 50, ...
            "MinLeafSize",  5, ...
            "CrossVal",     "off");

    case "svm"
        % Support Vector Machine with RBF kernel
        mdl = fitrsvm(XTrain, yTrain, ...
            "KernelFunction",   "rbf", ...
            "Standardize",      false, ...  % already z-scored
            "KernelScale",      "auto", ...
            "BoxConstraint",    1);

    case "gpr"
        % Gaussian Process Regression — provides uncertainty estimates
        mdl = fitrgp(XTrain, yTrain, ...
            "BasisFunction",    "constant", ...
            "KernelFunction",   "squaredexponential", ...
            "Standardize",      false, ...
            "FitMethod",        "sr", ...
            "PredictMethod",    "sr");

    case "linear"
        % Regularised linear regression (Lasso) — interpretable, fast
        mdl = fitrlinear(XTrain, yTrain, ...
            "Learner",    "leastsquares", ...
            "Regularization", "lasso", ...
            "Lambda",     "auto");

    otherwise
        error("RegressionLearner:unknownModel", ...
            "Unknown modelType ""%s"". Choose: ensemble, tree, svm, gpr, linear.", modelType);
end

fprintf("Training complete.\n\n");

%% Predictions

yTrainPred = predict(mdl, XTrain);
yValPred   = predict(mdl, XVal);
yTestPred  = predict(mdl, XTest);

%% Metrics Calculation

indTrain = computeMetrics(yTrainPred, yTrain, normParams, char(target));
indVal   = computeMetrics(yValPred,   yVal,   normParams, char(target));
indTest  = computeMetrics(yTestPred,  yTest,  normParams, char(target));

%% Model Saving

ts      = char(string(datetime("now", "Format", "yyyy_MM_dd_HH_mm_ss")));
netName = "reg_" + modelType + "_exog_" + ts;  % '_exog': only weather+time features

indicators = struct("Training", indTrain, "Validation", indVal, "Test", indTest);

regStruct = struct( ...
    "zoneId",     zoneId, ...
    "mdl",        mdl, ...
    "modelType",  modelType, ...
    "indicators", indicators, ...
    "normParams", normParams, ...
    "XTrain",     XTrain, ...
    "yTrain",     yTrain, ...
    "XVal",       XVal, ...
    "yVal",       yVal, ...
    "XTest",      XTest, ...
    "yTest",      yTest, ...
    "timeVectorTest", {timeVectorTest}, ...
    "config",     struct( ...
    "numLags",       numLags, ...
    "predictors",    predictors, ...
    "target",        target, ...
    "numInputFeatures", numInputFeatures ...
    ) ...
    );

models = struct();
models.(char(netName)) = regStruct;

saveModels(root, models, char(netName), []);

%% Final Summary in Command Window

cfg = regStruct.config;
ind = regStruct.indicators;

fprintf("\n========================================================\n");
fprintf("     REGRESSION MODEL SUMMARY — %s\n", netName);
fprintf("--------------------------------------------------------\n");
fprintf("   Zone:                %d\n",    zoneId);
fprintf("   Model Type:          %s\n",    modelType);
fprintf("   Features (%d):       %s\n",    numel(cfg.predictors), strjoin(cfg.predictors, ", "));
fprintf("   Lags (window):       %d samples (%.0f hours)\n", cfg.numLags, cfg.numLags / 2);
fprintf("   Input width:         %d columns\n", cfg.numInputFeatures);
fprintf("--------------------------------------------------------\n");
fprintf("   TRAINING METRICS (denormalized, actual kWh)\n");
fprintf("     R²:    %.4f\n",    ind.Training.RSquared);
fprintf("     RMSE:  %.4f kWh\n", ind.Training.RMSE);
fprintf("     MAE:   %.4f kWh\n", ind.Training.MAE);
fprintf("     MAPE:  %.2f %%\n",  ind.Training.MAPE);
fprintf("--------------------------------------------------------\n");
fprintf("   VALIDATION METRICS (denormalized, actual kWh)\n");
fprintf("     R²:    %.4f\n",    ind.Validation.RSquared);
fprintf("     RMSE:  %.4f kWh\n", ind.Validation.RMSE);
fprintf("     MAE:   %.4f kWh\n", ind.Validation.MAE);
fprintf("     MAPE:  %.2f %%\n",  ind.Validation.MAPE);
fprintf("--------------------------------------------------------\n");
fprintf("   TEST METRICS (denormalized, actual kWh)\n");
fprintf("     R²:    %.4f\n",    ind.Test.RSquared);
fprintf("     RMSE:  %.4f kWh\n", ind.Test.RMSE);
fprintf("     MAE:   %.4f kWh\n", ind.Test.MAE);
fprintf("     MAPE:  %.2f %%\n",  ind.Test.MAPE);
fprintf("========================================================\n\n");

%% Prediction Plots

% De-normalize for plotting
targetMu  = normParams.(char(target)).mean;
targetStd = normParams.(char(target)).std;

yTrainReal     = yTrain     .* targetStd + targetMu;
yValReal       = yVal       .* targetStd + targetMu;
yTestReal      = yTest      .* targetStd + targetMu;
trainPredReal  = yTrainPred .* targetStd + targetMu;
valPredReal    = yValPred   .* targetStd + targetMu;
testPredReal   = yTestPred  .* targetStd + targetMu;

% ── Time-series comparison: Real vs Predicted (Test Set, denormalized) ──
% Uses plotRegressionResults to draw day-separated lines with date labels on X.
plotRegressionResults(yTestPred, yTest, normParams, char(target), timeVectorTest);
title(sprintf("%s — Test Set: Real vs Predicted (denormalized kWh)", modelType), FontSize=13);

% ── Scatter: Real vs Predicted (Test Set) ────────────────────────────────
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

%% Residual Analysis (Test Set)

residuiTest = yTestReal - testPredReal;

figure("Name", "Residual Analysis — Test Set", "NumberTitle", "off", ...
    "Position", [100 100 1100 600]);
tiledlayout(2, 2, TileSpacing="compact", Padding="compact");

nexttile;
scatter(timeVectorTest, residuiTest, 20, [0.2 0.5 0.9], "filled", MarkerFaceAlpha=0.75);
yline(0, "--k", LineWidth=1);
title("Residuals Over Time", FontSize=12);
xlabel("Date"); ylabel("Error (kWh)");
grid on;

nexttile;
histogram(residuiTest, 30, FaceColor=[0.2 0.7 0.5], EdgeColor="none");
title("Residual Distribution", FontSize=12);
xlabel("Error (kWh)"); ylabel("Frequency");
grid on;

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

axQQ = nexttile;
qqplot(axQQ, residuiTest);
title(axQQ, "Q-Q Plot for Residuals vs Normal", FontSize=12);
grid(axQQ, "on");

sgtitle(sprintf("Residual Analysis — %s | Test Set (RMSE=%.2f kWh)", ...
    modelType, ind.Test.RMSE), FontSize=14, FontWeight="bold");

%% Hourly Error Breakdown (Test Set)

orarioTest      = hour(timeVectorTest);
erroreAssoluto  = abs(residuiTest);

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
title(sprintf("Hourly Error Breakdown — %s — Test Set", modelType), FontSize=13);
xticks(0:23);
xticklabels(string(0:23) + ":00");
xtickangle(45);
grid on; grid minor;

[~, oraMax] = max(errorePerOra);
fprintf("\nHour with maximum average error: %02d:00 (MAE = %.2f kWh)\n", ...
    oraMax - 1, errorePerOra(oraMax));
fprintf("Hour with minimum average error: %02d:00 (MAE = %.2f kWh)\n\n", ...
    find(errorePerOra == min(errorePerOra), 1) - 1, min(errorePerOra));

%% Persistence Baseline (y(t) = y(t-1))

yPers        = yTestReal(1:end-1);
yTestShifted = yTestReal(2:end);

ssResPers = sum((yTestShifted - yPers).^2);
ssTotPers = sum((yTestShifted - mean(yTestShifted)).^2);
r2Pers    = 1 - ssResPers / ssTotPers;
rmsePers  = sqrt(mean((yTestShifted - yPers).^2));
maePers   = mean(abs(yTestShifted - yPers));

fprintf("\n--- PERSISTENCE BASELINE COMPARISON ---\n");
fprintf("%-20s  %8s  %12s  %12s\n", "Model", "R^2", "RMSE (kWh)", "MAE (kWh)");
fprintf("%s\n", repmat('-', 1, 58));
fprintf("%-20s  %8.4f  %12.2f  %12.2f\n", modelType, ind.Test.RSquared, ind.Test.RMSE, ind.Test.MAE);
fprintf("%-20s  %8.4f  %12.2f  %12.2f\n", "Persistence",  r2Pers, rmsePers, maePers);
fprintf("%s\n", repmat('-', 1, 58));
