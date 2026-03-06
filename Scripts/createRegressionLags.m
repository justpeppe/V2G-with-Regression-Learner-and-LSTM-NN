function dataWithRegressors = createRegressionLags(inputData, regressors, predictorNames, targetName)
% createRegressionLags Creates lagged variables for training a Regression Learner model.
arguments
    inputData table
    regressors (1,1) double {mustBeInteger, mustBePositive}
    predictorNames
    targetName
end

if isstring(predictorNames)
    predictorNames = cellstr(predictorNames);
elseif ischar(predictorNames)
    predictorNames = {predictorNames};
end

if isstring(targetName)
    targetName = char(targetName);
end

if ~ismember("time_vector", inputData.Properties.VariableNames)
    error("The table must contain the time_vector variable.");
end

if ~isdatetime(inputData.time_vector)
    error("table.time_vector must be a datetime array.");
end

% Determine the unique days present in the dataset
daysArr = dateshift(inputData.time_vector, "start", "day");
uniqueDays = sort(unique(daysArr));
fprintf("Total days: %d\n", height(uniqueDays));

% Identify days that are strictly consecutive to the previous day in the dataset.
% This is crucial because lagging features across gaps in time (e.g., missing days)
% would create invalid training sequences.
validConsecutiveDays = false(height(uniqueDays), 1);
for idxDay = 2:height(uniqueDays)
    if days(uniqueDays(idxDay) - uniqueDays(idxDay - 1)) == 1
        validConsecutiveDays(idxDay) = true;
    end
end

numValidDays = sum(validConsecutiveDays);
fprintf("Valid consecutive days: %d\n", numValidDays);

% Calculate output matrix dimensions
numPredictors = numel(predictorNames);
numColumns = numPredictors * regressors + 1; % +1 for the target variable
estimatedRows = height(inputData);

% Pre-allocate the memory for the output matrix to optimize performance
dataMatrix = NaN(estimatedRows, numColumns);
timeVectorOut = NaT(estimatedRows, 1);
idxRowOut = 1;

% Generate descriptive column names for the lagged features (e.g., temp_t_1)
columnNames = cell(1, numColumns + 1);
colIdx = 1;
for lag = regressors:-1:1
    for idxPred = 1:numPredictors
        columnNames{colIdx} = sprintf("%s_t_%d", predictorNames{idxPred}, lag);
        colIdx = colIdx + 1;
    end
end
columnNames{colIdx} = targetName;
columnNames{end} = "time_vector";

% Process only valid consecutive days to ensure data continuity
for idxDay = find(validConsecutiveDays)'
    currentDay = uniqueDays(idxDay);
    previousDay = uniqueDays(idxDay - 1);

    % Extract data slices for the current and previous day
    idxCurrentDay = daysArr == currentDay;
    idxPreviousDay = daysArr == previousDay;

    previousData = inputData(idxPreviousDay, :);
    currentData = inputData(idxCurrentDay, :);

    % Concatenate the two days to allow for lagging features from the previous day into the current day
    concatTable = [previousData; currentData];
    numTotalSamples = height(concatTable);

    % Iterate through the joined days, starting after the initial lag period
    for idxSample = (regressors + 1):numTotalSamples
        colIdxInner = 1;

        % Populate the lagged regressors (t-N, t-(N-1), ..., t-1)
        for lag = regressors:-1:1
            idxLag = idxSample - lag;
            for idxPred = 1:numPredictors
                predName = predictorNames{idxPred};
                dataMatrix(idxRowOut, colIdxInner) = concatTable.(predName)(idxLag);
                colIdxInner = colIdxInner + 1;
            end
        end

        % Assign the target variable (t=0) and corresponding timestamp
        dataMatrix(idxRowOut, end) = concatTable.(targetName)(idxSample);
        timeVectorOut(idxRowOut) = concatTable.time_vector(idxSample);
        idxRowOut = idxRowOut + 1;
    end
end

% Trim the pre-allocated matrix to remove unused rows
dataMatrix = dataMatrix(1:idxRowOut-1, :);
timeVectorOut = timeVectorOut(1:idxRowOut-1);

% Convert back to a table and attach the timestamps
dataWithRegressors = array2table(dataMatrix, "VariableNames", columnNames(1:end-1));
dataWithRegressors.time_vector = timeVectorOut;

fprintf("Total rows created: %d\n", height(dataWithRegressors));
end
