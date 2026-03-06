%% Clean

clear
%% Data Loading

root = fileparts(mfilename('fullpath'));
addpath(root + "\Scripts");

zoneId = 9; % Zone 214 Trieste
datas = loadZoneData(root, zoneId);

%% Constants

numLags = 48;

% Predittori: meteo, festività, orario (seno/coseno per la ciclicità dell'ora e del giorno)
% e il consumo energetico storico (AAC_energy).
% Le feature orarie impediscono alla rete di limitarsi a copiare l'ultimo valore:
% sa «che ore sono», quindi può anticipare i picchi senza imbrogliare.
predictors = ["AAC_energy", "precipprob", "temp", "windspeed", "holiday_indicator", ...
    "hour_sin", "hour_cos", "day_sin", "day_cos"];
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

% Estraiamo le feature temporali esplicite (seno e coseno per ciclicità)
datas.hour_sin = sin(2 * pi * hour(datas.time_vector) / 24);
datas.hour_cos = cos(2 * pi * hour(datas.time_vector) / 24);
datas.day_sin  = sin(2 * pi * day(datas.time_vector, 'dayofweek') / 7);
datas.day_cos  = cos(2 * pi * day(datas.time_vector, 'dayofweek') / 7);

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

% --- Architettura della rete LSTM ---
% 1. sequenceInputLayer  → riceve le sequenze [numLags × numFeatures]
% 2. lstmLayer           → impara dipendenze temporali, outputMode="last" restituisce
%                          solo lo stato finale (many-to-one regression)
% 3. dropoutLayer        → disattiva casualmente il 20% dei neuroni → evita overfitting
% 4. fullyConnectedLayer → strato denso intermedio per combinare le feature LSTM
% 5. reluLayer           → attivazione non-lineare (taglia i valori negativi)
% 6. fullyConnectedLayer → strato di uscita: produce il singolo valore predetto
layers = [
    sequenceInputLayer(numFeatures, Normalization="none")
    lstmLayer(numHiddenUnits, OutputMode="last")
    dropoutLayer(drop)
    fullyConnectedLayer(numHiddenUnits / 2)
    reluLayer
    dropoutLayer(drop)
    fullyConnectedLayer(numResponses)
    ];

% --- Opzioni di addestramento ---
% Adam: ottimizzatore adattivo, ottimo per reti LSTM, converge rapidamente.
% Shuffle="every-epoch": rimescola le sequenze ad ogni epoca → la rete non
%   "memorizza" l'ordine stagionale ma generalizza i pattern.
% ValidationPatience=15: ferma l'addestramento se la validation loss non
%   migliora per 15 valutazioni consecutive (early stopping).
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

%% Salvataggio del Modello
% La struct contiene tutto il necessario per riprodurre e valutare la rete:
%   - net:         la rete addestrata (usa per fare predizioni con minibatchpredict)
%   - indicators:  metriche di training e validation (RMSE, R², MAPE, MAE, MSE)
%   - normParams:  parametri z-score per de-normalizzare le predizioni
%   - x/yTrain|Validation|Test: sequenze  e target pronti per predizioni future
%   - timeVector*: timestamp corrispondenti a ogni sequenza (per il plot)
%   - config:      parametri di configurazione della rete (hidden units, features, etc.)

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

%% Predizione e Plot dei Risultati
% Genera le predizioni de-normalizzate su training, validation e test
% e le plotta a confronto con i dati reali.

m = models.(netName); % alias per brevità

trainingPrediction   = minibatchpredict(m.net, m.xTrain);
validationPrediction = minibatchpredict(m.net, m.xValidation);
testPrediction       = minibatchpredict(m.net, m.xTest);

trainingFig   = plotResults(trainingPrediction,   m.yTrain,      m.normParams, target, m.timeVectorTrain);
validationFig = plotResults(validationPrediction, m.yValidation, m.normParams, target, m.timeVectorValidation);
testFig       = plotResults(testPrediction,        m.yTest,       m.normParams, target, m.timeVectorTest);

%% Calcolo Metriche sul Test Set (de-normalizzate, in kWh reali)
% computeMetrics calcola RMSE, R², MAPE, MAE, MSE de-normalizzando prima.
models.(netName).indicators.Test = computeMetrics(testPrediction, yTest, normParams, char(target));

%% Salvataggio Finale (unico salvataggio, include le metriche del test)
saveModels(root, models, char(netName), []);

%% Riepilogo Finale in Command Window
ind = models.(netName).indicators;
cfg = models.(netName).config;

fprintf("\n╔══════════════════════════════════════════════════════╗\n");
fprintf("║         LSTM MODEL SUMMARY — %s\n", netName);
fprintf("╠══════════════════════════════════════════════════════╣\n");
fprintf("║  Zona:             %d\n",      zoneId);
fprintf("║  Features (%d):   %s\n",      cfg.numFeatures, strjoin(cfg.predictors, ", "));
fprintf("║  Target:           %s\n",      cfg.target);
fprintf("║  Lags (finestra):  %d campioni\n", cfg.numLags);
fprintf("║  Hidden units:     %d (L1)  +  %d (L2)\n", cfg.numHiddenUnits, cfg.numHiddenUnits/2);
fprintf("╠══════════════════════════════════════════════════════╣\n");
fprintf("║  TRAINING METRICS (normalizzate, z-score)\n");
fprintf("║    R²:    %.4f\n",    ind.Training.Rsquared);
fprintf("║    RMSE:  %.4f\n",    ind.Training.RMSE);
fprintf("║    MAE:   %.4f\n",    ind.Training.MAE);
fprintf("║    MAPE:  %.2f %%\n", ind.Training.MAPE);
fprintf("╠══════════════════════════════════════════════════════╣\n");
fprintf("║  VALIDATION METRICS (normalizzate, z-score)\n");
fprintf("║    R²:    %.4f\n",    ind.Validation.Rsquared);
fprintf("║    RMSE:  %.4f\n",    ind.Validation.RMSE);
fprintf("║    MAE:   %.4f\n",    ind.Validation.MAE);
fprintf("║    MAPE:  %.2f %%\n", ind.Validation.MAPE);
fprintf("╠══════════════════════════════════════════════════════╣\n");
fprintf("║  TEST METRICS (de-normalizzate, in kWh reali)\n");
fprintf("║    R²:    %.4f\n",      ind.Test.RSquared);
fprintf("║    RMSE:  %.4f kWh\n",  ind.Test.RMSE);
fprintf("║    MAE:   %.4f kWh\n",  ind.Test.MAE);
fprintf("║    MAPE:  %.2f %%\n",   ind.Test.MAPE);
fprintf("╚══════════════════════════════════════════════════════╝\n\n");