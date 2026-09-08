function [trainData, valData, testData, normParams] = splitZoneData(data, windowSize, predictorNames, targetName)
% splitZoneData Splits time series data into train, val, and test sets.
% Generates sliding windows (windowSize timesteps) to predict the next single instant (t+1).

arguments
    data table
    windowSize (1,1) double {mustBePositive, mustBeInteger} = 48
    predictorNames string = string.empty
    targetName string = "AAC_energy"
end

%% 1. Check predictor and target variables
if isempty(predictorNames)
    isNumeric = varfun(@(col) isa(col, 'double'), data, 'OutputFormat', 'uniform');
    predictorNames = string(data.Properties.VariableNames(isNumeric));
else
    predictorNames = string(predictorNames);
end

targetCol = char(targetName);
featureCols = cellstr(predictorNames);

%% 2. Group days into seasonal clusters
data.day = dateshift(data.datetime, "start", "day");
allDays = unique(data.day);
dayCounts = groupcounts(data.day);
completeDays = allDays(dayCounts == windowSize);

% A gap greater than 1 day marks a new cluster
dayGaps = days(diff(completeDays));
clusterId = ones(size(completeDays));
clusterId(2:end) = clusterId(2:end) + cumsum(dayGaps > 1);
numClusters = max(clusterId);

%% 3. Assign train, validation, and test labels
% The last 4 days of each cluster are held out:
% Odd clusters tail -> validation, Even clusters tail -> test
daySplit = strings(size(completeDays));
daySplit(:) = "train";

for c = 1:numClusters
    clusterDays = completeDays(clusterId == c);
    tailDays = clusterDays(end-3:end);
    if mod(c, 2) == 1
        daySplit(ismember(completeDays, tailDays)) = "val";
    else
        daySplit(ismember(completeDays, tailDays)) = "test";
    end
end

% Assign clusterId and splitLabel to each row in the dataset
data.clusterId = zeros(height(data), 1);
data.splitLabel = strings(height(data), 1);
for i = 1:numel(completeDays)
    mask = data.day == completeDays(i);
    data.clusterId(mask) = clusterId(i);
    data.splitLabel(mask) = daySplit(i);
end

%% 4. Compute normalization parameters (on training rows only)
isTrain = data.splitLabel == "train";
XTrainRaw = data{isTrain, featureCols};
TTrainRaw = data.(targetCol)(isTrain);

normParams.predictorNames = predictorNames;
normParams.targetName = targetName;
normParams.mu = mean(XTrainRaw, 1);
normParams.sigma = std(XTrainRaw, 0, 1);
normParams.sigma(normParams.sigma == 0) = 1;

normParams.muTarget = mean(TTrainRaw);
normParams.sigmaTarget = std(TTrainRaw);
if normParams.sigmaTarget == 0
    normParams.sigmaTarget = 1;
end

% Normalize all predictors and target across the entire table
XNorm = (data{:, featureCols} - normParams.mu) ./ normParams.sigma;
TNorm = (data.(targetCol) - normParams.muTarget) ./ normParams.sigmaTarget;

%% 5. Extract sliding windows (Sequence-to-Scalar)
trainData = struct('X', {{}}, 'T', zeros(0,1), 'timestamps', datetime.empty(0,1));
valData   = struct('X', {{}}, 'T', zeros(0,1), 'timestamps', datetime.empty(0,1));
testData  = struct('X', {{}}, 'T', zeros(0,1), 'timestamps', datetime.empty(0,1));

for c = 1:numClusters
    clusterRows = find(data.clusterId == c);
    
    for k = windowSize + 1 : numel(clusterRows)
        targetRow = clusterRows(k);
        windowRows = clusterRows(k - windowSize : k - 1);
        
        split = data.splitLabel(targetRow);
        if split == "train"
            trainData.X{end+1, 1} = XNorm(windowRows, :);
            trainData.T(end+1, 1) = TNorm(targetRow);
            trainData.timestamps(end+1, 1) = data.datetime(targetRow);
        elseif split == "val"
            valData.X{end+1, 1} = XNorm(windowRows, :);
            valData.T(end+1, 1) = TNorm(targetRow);
            valData.timestamps(end+1, 1) = data.datetime(targetRow);
        elseif split == "test"
            testData.X{end+1, 1} = XNorm(windowRows, :);
            testData.T(end+1, 1) = TNorm(targetRow);
            testData.timestamps(end+1, 1) = data.datetime(targetRow);
        end
    end
end

%% 6. Print simple summary
fprintf("Split completed: %d train, %d val, %d test samples.\n", ...
    numel(trainData.T), numel(valData.T), numel(testData.T));

end
