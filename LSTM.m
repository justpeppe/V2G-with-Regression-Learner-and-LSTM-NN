%% Clean

clear
%% Data Loading

root = fileparts(mfilename('fullpath'));
addpath(root + "\Scripts");

zoneId = 9; % Zone 214 Trieste
datas = loadZoneData(root, zoneId);

%% Constants

numLags = 48;
predictors = ["AAC_energy", "precipprob", "temp", "windspeed", "holiday_indicator"];
target = "AAC_energy";
columnsToNormalize = unique([predictors, target], "stable");

validationDays = [
    datetime(2023, 2, 21); % Regressor
    datetime(2023, 2, 22); % Test day
    datetime(2023, 7, 21); % Regressor
    datetime(2023, 7, 22)  % Test day
    ];

testDays = [
    datetime(2023, 6, 7);  % Regressor
    datetime(2023, 6, 8);  % Test day
    datetime(2023, 10, 7); % Regressor
    datetime(2023, 10, 8)  % Test day
    ];

%% Data Preparation

[training, validation, test] = splitTrainValTest(datas, validationDays, testDays);

[trainingNorm, testNorm, normParams] = normalizeZScore(training, test, columnsToNormalize);

% BUG #4 FIX: include numLags context rows from training so that createLstmSequences
% can build valid sequences starting from the very first validation/test day.
% Without this, the first ~numLags samples of each split would be discarded.
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

fprintf("\n=== Sequence Diagnostics ===\n");
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
fprintf("===========================\n\n");


%% Long Short-Term Memory NN Model

numFeatures = size(xTrain{1}, 2);
fprintf("numFeatures:   %d\n", numFeatures);
numResponses = size(yTrain, 2);

numHiddenUnits = 128;
drop = 0.2;
mb = 64;
valFreq = max(1, ceil(numel(xTrain) / mb));

% BUG #3 FIX: simplified architecture – single LSTM with OutputMode="last" is
% more appropriate for many-to-one regression and far less prone to overfitting.
layers = [
    sequenceInputLayer(numFeatures, Normalization="none")
    lstmLayer(numHiddenUnits, OutputMode="last")
    dropoutLayer(drop)
    fullyConnectedLayer(numHiddenUnits / 2)
    reluLayer
    dropoutLayer(drop)
    fullyConnectedLayer(numResponses)
    ];

options = trainingOptions("adam", ...
    MaxEpochs=150, ...
    MiniBatchSize=mb, ...
    Shuffle="never", ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=50, ...     % BUG #6 FIX: less aggressive LR decay (was 30)
    GradientThreshold=1, ...
    GradientThresholdMethod="l2norm", ...
    L2Regularization=1e-4, ...
    ValidationData={xValidation, yValidation}, ...
    ValidationFrequency=valFreq, ...
    ValidationPatience=15, ...      % BUG #8 FIX: early stopping if val loss stagnates
    Metrics=["rsquared", "rmse", "mape", "mae", "mse"], ...
    Plots="training-progress", ...
    Verbose=false, ...
    ExecutionEnvironment="auto" ...
    );

[net, info] = trainnet(xTrain, yTrain, layers, "mse", options);

%% Saving Data

ts = char(string(datetime("now", "Format", "yyyy_MM_dd_HH_mm_ss")));
netName = "net_" + ts;

indicators = getBestIndicators(info);

% Using structural initializations per standards
netStruct = struct( ...
    "zoneId", zoneId, ...
    "net", net, ...
    "net_indicators", indicators, ...
    "layers", layers, ...
    "options", options, ...
    "numFeatures", numFeatures, ...
    "numResponses", numResponses, ...
    "numHiddenUnits", numHiddenUnits, ...
    "xTrain", {xTrain}, ...
    "yTrain", yTrain, ...
    "timeVectorTrain", timeVectorTrain, ...
    "xTest", {xTest}, ...
    "yTest", yTest, ...
    "timeVectorTest", timeVectorTest, ...
    "xValidation", {xValidation}, ...
    "yValidation", yValidation, ...
    "timeVectorValidation", timeVectorValidation, ...
    "trainingData", training, ...
    "testData", test, ...
    "validationData", validation, ...
    "trainingDataNorm", trainingNorm, ...
    "testDataNorm", testNorm, ...
    "validationDataNorm", validationWithCtx, ...
    "normParams", normParams ...
    );

models = struct();
models.(netName) = netStruct;

saveModels(root, models, char(netName), net);

%% LSTM NN Plot

trainingPrediction = minibatchpredict(models.(netName).net, models.(netName).xTrain);
trainingFig = plotResults(trainingPrediction, models.(netName).yTrain, models.(netName).normParams, target, models.(netName).timeVectorTrain);

validationPrediction = minibatchpredict(models.(netName).net, models.(netName).xValidation);
validationFig = plotResults(validationPrediction, models.(netName).yValidation, models.(netName).normParams, target, models.(netName).timeVectorValidation);

testPrediction = minibatchpredict(models.(netName).net, models.(netName).xTest);
testFig = plotResults(testPrediction, models.(netName).yTest, models.(netName).normParams, target, models.(netName).timeVectorTest);

models.(netName).net_indicators.Test = computeMetrics(testPrediction, yTest, normParams, char(target));
saveModels(root, models, char(netName), net);

currentDate = char(string(datetime("now", "Format", "yyyy_MM_dd")));
matFile = fullfile(string(root), "Sessioni", string(currentDate), "Models_" + string(currentDate) + ".mat");