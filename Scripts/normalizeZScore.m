function [trainingNorm, testNorm, normParams] = normalizeZScore(trainingSet, testSet, columnsToNormalize)
% normalizeZScore Normalizes specific columns using z-score based on training parameters.
arguments
    trainingSet {mustBeA(trainingSet, ["table", "numeric"])}
    testSet {mustBeA(testSet, ["table", "numeric"])}
    columnsToNormalize
end

isTableInput = istable(trainingSet);

if isTableInput
    if ischar(columnsToNormalize)
        columnsToNormalize = {columnsToNormalize};
    elseif isstring(columnsToNormalize)
        columnsToNormalize = cellstr(columnsToNormalize);
    end
end

normParams = struct();
if isTableInput
    trainingNorm = table();
    testNorm = table();
else
    numCols = numel(columnsToNormalize);
    trainingNorm = zeros(size(trainingSet, 1), numCols);
    testNorm = zeros(size(testSet, 1), numCols);
end

for i = 1:numel(columnsToNormalize)
    if isTableInput
        colName = columnsToNormalize{i};
        vTrain = trainingSet.(colName);
        vTest = testSet.(colName);
    else
        colIdx = columnsToNormalize(i);
        vTrain = trainingSet(:, colIdx);
        vTest = testSet(:, colIdx);
    end

    if isdatetime(vTrain)
        mu = mean(vTrain, "omitnat");
        sg = std(vTrain, 0, "omitnat");
        sgIsZero = (seconds(sg) == 0) | isnan(seconds(sg));
        if sgIsZero
            sg = seconds(1);
        end
        zTrain = (vTrain - mu) ./ sg;
        zTest = (vTest - mu) ./ sg;
    elseif isduration(vTrain)
        mu = mean(vTrain, "omitnan");
        sg = std(vTrain, 0, "omitnan");
        sgIsZero = (seconds(sg) == 0) | isnan(seconds(sg));
        if sgIsZero
            sg = seconds(1);
        end
        zTrain = (vTrain - mu) ./ sg;
        zTest = (vTest - mu) ./ sg;
    else
        mu = mean(vTrain, "omitnan");
        sg = std(vTrain, 0, "omitnan");
        if isnan(sg) || sg == 0
            sg = 1;
        end
        zTrain = (vTrain - mu) ./ sg;
        zTest = (vTest - mu) ./ sg;
    end

    if isTableInput
        trainingNorm.(colName) = zTrain;
        testNorm.(colName) = zTest;
        safeField = matlab.lang.makeValidName(colName);
        normParams.(safeField).mean = mu;
        normParams.(safeField).std = sg;
    else
        trainingNorm(:, i) = zTrain;
        testNorm(:, i) = zTest;
        pname = sprintf("colonna_%d", colIdx);
        normParams.(pname).mean = mu;
        normParams.(pname).std = sg;
    end
end

if isTableInput && ismember("time_vector", trainingSet.Properties.VariableNames) && ismember("time_vector", testSet.Properties.VariableNames)
    trainingNorm.time_vector = trainingSet.time_vector;
    testNorm.time_vector = testSet.time_vector;
end

fprintf("Tables normalized with z-score method.\n");
end
