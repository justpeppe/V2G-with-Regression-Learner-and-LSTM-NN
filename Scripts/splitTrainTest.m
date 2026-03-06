function [training, test] = splitTrainTest(inputTable, testDays)
% splitTrainTest Splits the input table into training and test sets based on specific days.
arguments
    inputTable table
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

% Find which rows in the main table belong to the designated test days
idxTest = ismember(tempDays, testDays);

% Split the dataset:
% Training set gets everything that is NOT a test day
training = inputTable(~idxTest, :);
% Test set gets only the specific assigned test days
test = inputTable(idxTest, :);

fprintf("Created Training and Test tables for Regression Learner.\n");
end