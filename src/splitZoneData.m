function [trainSet, valSet, testSet, normParams] = splitZoneData(data, windowSize, predictorNames, targetName)
% splitZoneData Build train/val/test LSTM windows (day T-1 -> day T energy).
arguments
    data table
    windowSize (1,1) double {mustBePositive, mustBeInteger} = 48
    predictorNames string = string.empty
    targetName string = "AAC_energy"
end

TAIL_DAYS_PER_CLUSTER = 4;
HOLIDAY_THRESHOLD = 0.99;

data.day = dateshift(data.datetime, "start", "day");
allDays = unique(data.day);

samplesPerDay = zeros(size(allDays));
for i = 1:numel(allDays)
    samplesPerDay(i) = sum(data.day == allDays(i));
end
completeDays = allDays(samplesPerDay == windowSize);

% cluster = run of consecutive complete days; a gap of months starts a new one
if numel(completeDays) < 2
    clusterId = ones(size(completeDays));
else
    dayGaps = days(diff(completeDays));
    clusterId = ones(size(completeDays));
    clusterId(2:end) = clusterId(2:end) + cumsum(dayGaps > 1);
end
numClusters = max(clusterId);

% a day is usable only if T-1 is also complete (LSTM needs a full input day)
isUsable = false(size(completeDays));
for i = 1:numel(completeDays)
    isUsable(i) = any(completeDays == completeDays(i) - days(1));
end
usableDays = completeDays(isUsable);
usableClusterId = clusterId(isUsable);

% last TAIL_DAYS_PER_CLUSTER days of each cluster held out: odd -> val, even -> test
splitLabel = strings(size(usableDays));
splitLabel(:) = "train";
for c = 1:numClusters
    clusterDays = usableDays(usableClusterId == c);
    if numel(clusterDays) < TAIL_DAYS_PER_CLUSTER
        continue
    end
    tailDays = clusterDays(end - TAIL_DAYS_PER_CLUSTER + 1:end);
    if mod(c, 2) == 1
        splitLabel(ismember(usableDays, tailDays)) = "val";
    else
        splitLabel(ismember(usableDays, tailDays)) = "test";
    end
end

% determine feature names
if ~isempty(predictorNames)
    % accept string or cell input; ensure they exist in table
    predictorNames = string(predictorNames);
    missing = setdiff(cellstr(predictorNames), data.Properties.VariableNames);
    if ~isempty(missing)
        error("splitZoneData:MissingPredictors", "Predictor(s) not found in data: %s", strjoin(missing, ", "));
    end
    featureNames = cellstr(predictorNames);
else
    isDouble = varfun(@(col) isa(col, 'double'), data, 'OutputFormat', 'uniform');
    featureNames = data.Properties.VariableNames(isDouble); % skip non-numeric columns
end

% prepare sets
trainSet.X = {}; trainSet.Y = {}; trainSet.days = datetime.empty(0, 1); trainSet.isNonWorking = logical.empty(0, 1);
valSet = trainSet;
testSet = trainSet;

for i = 1:numel(usableDays)
    targetDay = usableDays(i);
    inputRows = sortrows(data(data.day == targetDay - days(1), :), "datetime");
    targetRows = sortrows(data(data.day == targetDay, :), "datetime");

    % check row counts
    if height(inputRows) ~= windowSize || height(targetRows) ~= windowSize
        continue
    end

    X = inputRows{:, featureNames};
    if ~ismember(cellstr(targetName), data.Properties.VariableNames)
        error("splitZoneData:MissingTarget", "Target variable '%s' not found in data.", targetName);
    end
    Y = targetRows.(char(targetName));
    isNonWorking = targetRows.holiday_proximity(1) > HOLIDAY_THRESHOLD;

    if splitLabel(i) == "train"
        trainSet.X{end+1} = X; trainSet.Y{end+1} = Y;
        trainSet.days(end+1,1) = targetDay; trainSet.isNonWorking(end+1,1) = isNonWorking;
    elseif splitLabel(i) == "val"
        valSet.X{end+1} = X; valSet.Y{end+1} = Y;
        valSet.days(end+1,1) = targetDay; valSet.isNonWorking(end+1,1) = isNonWorking;
    else
        testSet.X{end+1} = X; testSet.Y{end+1} = Y;
        testSet.days(end+1,1) = targetDay; testSet.isNonWorking(end+1,1) = isNonWorking;
    end
end

% If no training samples, warn and return empty sets
if isempty(trainSet.X)
    warning("splitZoneData:NoTrainSamples", "No training samples found. Returning empty sets.");
    normParams.featureNames = featureNames;
    normParams.mu = [];
    normParams.sigma = [];
    return
end

% scaler fit on training predictors only, then applied to all three sets
trainMatrix = cat(1, trainSet.X{:});
normParams.featureNames = featureNames;
normParams.mu = mean(trainMatrix, 1);
normParams.sigma = std(trainMatrix, 0, 1);
normParams.sigma(normParams.sigma == 0) = 1;

for i = 1:numel(trainSet.X)
    trainSet.X{i} = (trainSet.X{i} - normParams.mu) ./ normParams.sigma;
end
for i = 1:numel(valSet.X)
    valSet.X{i} = (valSet.X{i} - normParams.mu) ./ normParams.sigma;
end
for i = 1:numel(testSet.X)
    testSet.X{i} = (testSet.X{i} - normParams.mu) ./ normParams.sigma;
end

sets = {trainSet, valSet, testSet};
labels = ["training", "validation", "test"];
for k = 1:3
    s = sets{k};
    if isempty(s.days)
        fprintf("The %s set is empty.\n", labels(k));
        continue
    end
    nNonWorking = sum(s.isNonWorking);
    nWorking = numel(s.days) - nNonWorking;
    fprintf("The %s set has %d days (%d working, %d non-working).\n", ...
        labels(k), numel(s.days), nWorking, nNonWorking);

    listedDays = sort(s.days); listedDays.Format = "yyyy-MM-dd";
    fprintf("  Days: %s\n", strjoin(string(listedDays), ", "));
end

end
