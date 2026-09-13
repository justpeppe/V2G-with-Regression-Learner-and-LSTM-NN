function [trainTable, valTable, testTable, normParams] = prepareTabularData(data, windowSize, targetName)
% prepareTabularData Prepares tabular dataset with lag features for Regression Learner.
% Extracts domain-informed lag predictors and rolling statistics aligned with splitZoneData.

arguments
    data table
    windowSize (1,1) double {mustBePositive, mustBeInteger} = 48
    targetName (1,1) string = "AAC_energy"
end

%% Group days into seasonal clusters
data.day = dateshift(data.datetime, "start", "day");
allDays = unique(data.day);
dayCounts = groupcounts(data.day);
completeDays = allDays(dayCounts == windowSize);

% A gap greater than 1 day marks a new cluster
dayGaps = days(diff(completeDays));
clusterId = ones(size(completeDays));
clusterId(2:end) = clusterId(2:end) + cumsum(dayGaps > 1);
numClusters = max(clusterId);

%% Assign train, validation, and test labels
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

% Assign clusterId and splitLabel to each row
data.clusterId = zeros(height(data), 1);
data.splitLabel = strings(height(data), 1);
for i = 1:numel(completeDays)
    mask = data.day == completeDays(i);
    data.clusterId(mask) = clusterId(i);
    data.splitLabel(mask) = daySplit(i);
end

%% Extract tabular features cluster by cluster
% Ensures lag features and rolling statistics never cross cluster boundaries
featureTables = cell(numClusters, 1);

for c = 1:numClusters
    cRows = find(data.clusterId == c);
    nCRows = numel(cRows);
    
    if nCRows <= windowSize
        continue;
    end
    
    validCRows = cRows(windowSize + 1 : end);
    numSamples = numel(validCRows);
    
    lag1 = zeros(numSamples, 1);
    lag2 = zeros(numSamples, 1);
    lag48 = zeros(numSamples, 1);
    rollMean24h = zeros(numSamples, 1);
    rollStd24h = zeros(numSamples, 1);
    
    energyVec = data.(targetName)(cRows);
    
    for k = 1:numSamples
        idxInCluster = windowSize + k;
        lag1(k) = energyVec(idxInCluster - 1);
        lag2(k) = energyVec(idxInCluster - 2);
        lag48(k) = energyVec(idxInCluster - windowSize);
        pastWindow = energyVec(idxInCluster - windowSize : idxInCluster - 1);
        rollMean24h(k) = mean(pastWindow);
        rollStd24h(k) = std(pastWindow);
    end
    
    subT = table();
    subT.datetime = data.datetime(validCRows);
    subT.lag1_energy = lag1;
    subT.lag2_energy = lag2;
    subT.lag48_energy = lag48;
    subT.rollMean24h = rollMean24h;
    subT.rollStd24h = rollStd24h;
    
    if ismember("temp", data.Properties.VariableNames)
        subT.temp = data.temp(validCRows);
    end
    if ismember("cosHour", data.Properties.VariableNames)
        subT.cosHour = data.cosHour(validCRows);
    end
    if ismember("sinHour", data.Properties.VariableNames)
        subT.sinHour = data.sinHour(validCRows);
    end
    if ismember("holiday_proximity", data.Properties.VariableNames)
        subT.holiday_proximity = data.holiday_proximity(validCRows);
    end
    
    subT.(targetName) = energyVec(windowSize + 1 : end);
    subT.splitLabel = data.splitLabel(validCRows);
    
    featureTables{c} = subT;
end

tabularData = vertcat(featureTables{:});

%% Partition into train, validation, and test tables
isTrain = tabularData.splitLabel == "train";
isVal   = tabularData.splitLabel == "val";
isTest  = tabularData.splitLabel == "test";

featureCols = ["lag1_energy", "lag2_energy", "lag48_energy", "rollMean24h", "rollStd24h"];
if ismember("temp", tabularData.Properties.VariableNames)
    featureCols(end+1) = "temp";
end
if ismember("cosHour", tabularData.Properties.VariableNames)
    featureCols(end+1) = "cosHour";
end
if ismember("sinHour", tabularData.Properties.VariableNames)
    featureCols(end+1) = "sinHour";
end
if ismember("holiday_proximity", tabularData.Properties.VariableNames)
    featureCols(end+1) = "holiday_proximity";
end

normParams.mu = mean(tabularData{isTrain, featureCols}, 1);
normParams.sigma = std(tabularData{isTrain, featureCols}, 0, 1);
normParams.sigma(normParams.sigma == 0) = 1;
normParams.muTarget = mean(tabularData.(targetName)(isTrain));
normParams.sigmaTarget = std(tabularData.(targetName)(isTrain));
if normParams.sigmaTarget == 0
    normParams.sigmaTarget = 1;
end
normParams.featureCols = featureCols;
normParams.targetName = targetName;

trainTable = removevars(tabularData(isTrain, :), "splitLabel");
valTable   = removevars(tabularData(isVal, :),   "splitLabel");
testTable  = removevars(tabularData(isTest, :),  "splitLabel");

fprintf("Tabular split completed: %d train, %d val, %d test samples.\n", ...
    height(trainTable), height(valTable), height(testTable));

end
