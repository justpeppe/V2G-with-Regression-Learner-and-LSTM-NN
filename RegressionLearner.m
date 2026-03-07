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
% Identical pipeline to LSTM.m (same zone, configurable predictors, same split).
% The feature set can be dynamically configured via the 'useAutoregressive' flag:
%   1. Autoregressive Data: AAC_energy (historical energy consumption) — OPTIONAL.
%   2. Weather Data: precipprob, temp, windspeed.
%   3. Calendar Data: holiday_indicator.
%   4. Cyclical Time: hour_sin, hour_cos, day_sin, day_cos.
%
% NAMING CONVENTION
% Models are saved using standard-compliant lowerCamelCase naming strings
% (<= 32 characters) containing the type and 'HHMM' time (e.g., regTreeExog1105).
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
useAutoregressive = true; % Set to true to include AAC_energy in predictors, false for exogenous only

% Model type selector — choose one of:
%   "ensemble"  → Gradient Boosted Trees (fitrensemble, 'LSBoost')
%   "tree"      → Regression Tree       (fitrtree)
%   "svm"       → Support Vector Machine (fitrsvm, RBF kernel)
%   "gpr"       → Gaussian Process Regression (fitrgp)
%   "linear"    → Regularised Linear (fitrlinear, Lasso)
modelType = "ensemble";

% Exogenous predictors: weather + time
exogenousPredictors = ["precipprob", "temp", "windspeed", "holiday_indicator", ...
    "hour_sin", "hour_cos", "day_sin", "day_cos"];

if useAutoregressive
    predictors = ["AAC_energy", exogenousPredictors];
else
    predictors = exogenousPredictors;
end

target     = "AAC_energy";
columnsToNormalize = unique([predictors, target], "stable");

%% Automated Report Setup
currentTimeOrario  = char(string(datetime("now", "Format", "HHmm")));
if modelType == "svm"
    modelNameCamel = "Svm";
elseif modelType == "gpr"
    modelNameCamel = "Gpr";
else
    modelNameCamel = upper(extractBefore(modelType, 2)) + lower(extractAfter(modelType, 1));
end

if useAutoregressive
    netName = "reg" + modelNameCamel + "AutoReg" + currentTimeOrario;
else
    netName = "reg" + modelNameCamel + "Exog" + currentTimeOrario;
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

%% Target Autocorrelation
% The autocorrelation of AAC_energy justifies the choice of numLags=48:
% a significant coefficient at lag 48 indicates that the consumption
% 24 hours ago is highly predictive of the current value.

figACF_Target = figure("Name", "Target Autocorrelation", "NumberTitle", "off");
autocorr(yTrainReal, NumLags=100);
title("Autocorrelation of AAC\_energy (Training Set)", FontSize=13);
xlabel("Lag (30-min time steps)");
ylabel("Autocorrelation Coefficient");
xline(48, "--r", "Lag 48 (24h)", LabelVerticalAlignment="bottom");
grid on;
exportgraphics(figACF_Target, fullfile(reportDir, "Analysis_01_Target_ACF.png"), Resolution=300);
savefig(figACF_Target, fullfile(reportDir, "fig", "Analysis_01_Target_ACF.fig"));

% ── Time-series comparison: Real vs Predicted (Test Set, denormalized) ──
% Uses plotRegressionResults to draw day-separated lines with date labels on X.
figTimeSeries = plotRegressionResults(yTestPred, yTest, normParams, char(target), timeVectorTest);
title(sprintf("%s — Test Set: Real vs Predicted (denormalized kWh)", modelType), FontSize=13);
exportgraphics(figTimeSeries, fullfile(reportDir, "Analysis_01_TimeSeries.png"), Resolution=300);
savefig(figTimeSeries, fullfile(reportDir, "fig", "Analysis_01_TimeSeries.fig"));

% ── Scatter: Real vs Predicted (Test Set) ────────────────────────────────
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

exportgraphics(figScatter, fullfile(reportDir, "Analysis_02_Scatter.png"), Resolution=300);
savefig(figScatter, fullfile(reportDir, "fig", "Analysis_02_Scatter.fig"));

%% Residual Analysis (Test Set)

residuiTest = yTestReal - testPredReal;

figResid = figure("Name", "Residual Analysis — Test Set", "NumberTitle", "off", ...
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

exportgraphics(figResid, fullfile(reportDir, "Analysis_03_Residuals.png"), Resolution=300);
savefig(figResid, fullfile(reportDir, "fig", "Analysis_03_Residuals.fig"));

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
title(sprintf("Hourly Error Breakdown — %s — Test Set", modelType), FontSize=13);
xticks(0:23);
xticklabels(string(0:23) + ":00");
xtickangle(45);
grid on; grid minor;

exportgraphics(figHourly, fullfile(reportDir, "Analysis_04_HourlyError.png"), Resolution=300);
savefig(figHourly, fullfile(reportDir, "fig", "Analysis_04_HourlyError.fig"));

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
[~, idxOrd] = sort(erroreAssolutoTest, "descend");
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

diary off;
