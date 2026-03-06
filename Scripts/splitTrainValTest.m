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

tempDays = dateshift(inputTable.time_vector, "start", "day");
testDays = dateshift(testDays, "start", "day");
validationDays = dateshift(validationDays, "start", "day");

idxTest = ismember(tempDays, testDays);
idxValidation = ismember(tempDays, validationDays);

training = inputTable(~idxTest & ~idxValidation, :);
validation = inputTable(idxValidation, :);
test = inputTable(idxTest, :);

fprintf("Created Training, Validation, and Test tables for LSTM NN.\n");
end