function saveModels(rootFolder, models, varName, currentNet)
% saveModels Saves the trained model into a date-organized folder structure.
arguments
    rootFolder (1,1) string
    models struct
    varName (1,:) char
    currentNet
end

% Create a daily folder structure to organize saved models
currentDate = char(string(datetime("now", "Format", "yyyy_MM_dd")));
dateFolder = fullfile(string(rootFolder), "Sessioni", string(currentDate));

if ~exist(dateFolder, "dir")
    mkdir(dateFolder);
end

% Define the standardized mature file name for the day
matFile = fullfile(dateFolder, "Models_" + string(currentDate) + ".mat");

% Attach the trained network object if provided
if ~isempty(currentNet)
    models.(varName).net = currentNet;
end

% Load any existing models from today to append to them without overwriting
if exist(matFile, "file")
    loadedData = load(matFile, "models");
    if isfield(loadedData, "models") && isstruct(loadedData.models)
        existingStruct = loadedData.models;
    else
        existingStruct = struct();
    end
else
    existingStruct = struct();
end

% Append the new model to the struct
existingStruct.(varName) = models.(varName);
models = existingStruct;

% Save using v7.3 flag to support files larger than 2GB
save(matFile, "models", "-v7.3");
fprintf("Model ""%s"" saved in: %s\n", varName, matFile);
end
