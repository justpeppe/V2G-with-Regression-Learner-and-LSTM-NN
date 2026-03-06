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

tempDays = dateshift(inputTable.time_vector, "start", "day");
testDays = dateshift(testDays, "start", "day");

idxTest = ismember(tempDays, testDays);

training = inputTable(~idxTest, :);
test = inputTable(idxTest, :);

fprintf("Created Training and Test tables for Regression Learner.\n");
end