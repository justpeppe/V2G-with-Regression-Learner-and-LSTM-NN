function [X, Y, timeVectorFrames] = createLstmSequences(tbl, predictors, target)
% createLstmSequences Extracts contiguous sequences for Seq2Seq LSTM training.
% It splits continuous blocks into X (1:end-1) and Y (2:end).
arguments
    tbl table
    predictors
    target
end

if ischar(predictors)
    predictors = {predictors};
elseif isstring(predictors)
    predictors = cellstr(predictors);
end

if ischar(target)
    target = {target};
elseif isstring(target)
    target = cellstr(target);
end

if ~ismember("time_vector", tbl.Properties.VariableNames)
    error("The table must contain the time_vector variable.");
end

% Sort by time to ensure chronological order
tbl = sortrows(tbl, "time_vector");

% Find contiguous blocks based on 30-minute intervals
timeDiffs = diff(tbl.time_vector);
% A gap is any jump larger than the standard 30-minutes (we use 35 mins tolerance)
gapIndices = find(timeDiffs > minutes(35));

startIdx = 1;
breakPoints = [gapIndices; height(tbl)];

XTemp = cell(numel(breakPoints), 1);
YTemp = cell(numel(breakPoints), 1);
timeTemp = cell(numel(breakPoints), 1);

xMat = table2array(tbl(:, predictors));
yMat = table2array(tbl(:, target));

validCount = 0;
for i = 1:numel(breakPoints)
    endIdx = breakPoints(i);

    % A sequence requires at least 2 points to create a t and t+1 pair
    if (endIdx - startIdx) >= 1
        validCount = validCount + 1;

        sequenceX = xMat(startIdx:endIdx-1, :);
        sequenceY = yMat(startIdx+1:endIdx, :);
        sequenceTime = tbl.time_vector(startIdx+1:endIdx);

        XTemp{validCount} = sequenceX;
        YTemp{validCount} = sequenceY;
        timeTemp{validCount} = sequenceTime;
    end

    startIdx = endIdx + 1;
end

X = XTemp(1:validCount);
Y = YTemp(1:validCount);
timeVectorFrames = timeTemp(1:validCount);

fprintf("Number of Seq2Seq contiguous blocks created: %d\n", validCount);
end
