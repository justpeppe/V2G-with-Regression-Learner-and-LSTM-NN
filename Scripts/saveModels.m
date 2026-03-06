function saveModels(rootFolder, models, varName, currentNet)
% saveModels Saves the trained model into a date-organized folder structure.
arguments
    rootFolder (1,1) string
    models struct
    varName (1,:) char
    currentNet
end

currentDate = char(string(datetime("now", "Format", "yyyy_MM_dd")));
dateFolder = fullfile(string(rootFolder), "Sessioni", string(currentDate));

if ~exist(dateFolder, "dir")
    mkdir(dateFolder);
end

matFile = fullfile(dateFolder, "Models_" + string(currentDate) + ".mat");

if ~isempty(currentNet)
    models.(varName).net = currentNet;
end

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

existingStruct.(varName) = models.(varName);
models = existingStruct;

save(matFile, "models", "-v7.3");
fprintf("Model ""%s"" saved in: %s\n", varName, matFile);
end
