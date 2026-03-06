function [training, validation, test] = splitTrainValTest(inputTable, validationDays, testDays)
% splitTrainValTest Splits the input table into training, validation, and test sets based on specific days.
arguments
    inputTable table
    validationDays datetime
    testDays datetime
end

if ~ismember("time_vector", inputTable.Properties.VariableNames)
    error("The table must contain the time_vector variable.");
end

if ~isdatetime(inputTable.time_vector)
    error("table.time_vector must be a datetime array.");
end

% Extract just the date part (ignoring the time) for accurate day-level grouping
tempDays = dateshift(inputTable.time_vector, "start", "day");
testDays = dateshift(testDays, "start", "day");
validationDays = dateshift(validationDays, "start", "day");

% Find which rows in the main table belong to the designated validation and test days
idxTest = ismember(tempDays, testDays);
idxValidation = ismember(tempDays, validationDays);

% Split the dataset:
% Training set gets everything that is NOT a validation OR test day
training = inputTable(~idxTest & ~idxValidation, :);
% Validation set
validation = inputTable(idxValidation, :);
% Test set
test = inputTable(idxTest, :);

fprintf("Created Training, Validation, and Test tables for LSTM NN.\n");
end