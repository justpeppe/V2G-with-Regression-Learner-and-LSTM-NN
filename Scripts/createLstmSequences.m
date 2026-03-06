function [xTrain, yTrain, timeVectorLags] = createLstmSequences(tbl, numPredictors, columnsTable, target)
% createLstmSequences Extracts sequences of predictors and targets for LSTM training.
arguments
    tbl table
    numPredictors (1,1) double {mustBeInteger, mustBePositive}
    columnsTable
    target
end

if ischar(columnsTable)
    columnsTable = {columnsTable};
elseif isstring(columnsTable)
    columnsTable = cellstr(columnsTable);
end

if ischar(target)
    target = {target};
elseif isstring(target)
    target = cellstr(target);
end

if ~ismember("time_vector", tbl.Properties.VariableNames)
    error("The table must contain the time_vector variable.");
end

if ~isdatetime(tbl.time_vector)
    error("tbl.time_vector must be a datetime array.");
end

daysArr = dateshift(tbl.time_vector, "start", "day");
uniqueDays = sort(unique(daysArr));
fprintf("Total days: %d\n", height(uniqueDays));

validConsecutiveDays = false(height(uniqueDays), 1);
for idxDay = 2:height(uniqueDays)
    if days(uniqueDays(idxDay) - uniqueDays(idxDay - 1)) == 1
        validConsecutiveDays(idxDay) = true;
    end
end

numValidDays = sum(validConsecutiveDays);
fprintf("Valid consecutive days: %d\n", numValidDays);

xMat = table2array(tbl(:, columnsTable));
yMat = table2array(tbl(:, target));

numRows = size(tbl, 1);
numFeatures = size(xMat, 2);
numResponses = size(yMat, 2);

fprintf("Sequence length (numPredictors): %d\n", numPredictors);
fprintf("Number of features: %d\n", numFeatures);
fprintf("Number of targets: %d\n", numResponses);

xTrainTemp = cell(numRows, 1);
yTrainTemp = zeros(numRows, numResponses);
timeTemp = NaT(numRows, 1);
idxSequence = 1;

for i = 1:(numRows - numPredictors)
    tEnd = i + numPredictors - 1;
    tTarget = tEnd + 1;

    sequenceDays = daysArr(i:tTarget);
    uniqueSequenceDays = unique(sequenceDays);

    isValid = true;
    if numel(uniqueSequenceDays) > 1
        for j = 2:numel(uniqueSequenceDays)
            if days(uniqueSequenceDays(j) - uniqueSequenceDays(j-1)) ~= 1
                isValid = false;
                break;
            end
        end
    end

    if isValid
        xTrainTemp{idxSequence} = xMat(i:tEnd, :);  % [numLags × numFeatures] (Time x Channels format for trainnet)
        yTrainTemp(idxSequence, :) = yMat(tTarget, :);
        timeTemp(idxSequence, 1) = tbl.time_vector(tTarget);
        idxSequence = idxSequence + 1;
    end
end

lastValidIdx = idxSequence - 1;
xTrain = xTrainTemp(1:lastValidIdx);
yTrain = yTrainTemp(1:lastValidIdx, :);
timeVectorLags = timeTemp(1:lastValidIdx, 1);

fprintf("Number of sequences created: %d\n", lastValidIdx);
if lastValidIdx > 0
    seqShape = size(xTrain{1});
    fprintf("  Sequence shape xTrain{1}: [%d x %d]  (expected [numLags x numFeatures] = [%d x %d])\n", ...
        seqShape(1), seqShape(2), numPredictors, numFeatures);
    fprintf("  yTrain shape:             [%d x %d]\n", size(yTrain, 1), size(yTrain, 2));
else
    warning("createLstmSequences: no valid sequences created — check data continuity and numPredictors.");
end
fprintf("LSTM sequence creation completed.\n");
end
