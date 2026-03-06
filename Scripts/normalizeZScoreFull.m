function [datasNorm, normParams] = normalizeZScoreFull(datas, columnsToNormalize)
% normalizeZScoreFull Normalizes specific columns using z-score for the full dataset.
arguments
    datas table
    columnsToNormalize
end

if isstring(columnsToNormalize)
    columnsToNormalize = cellstr(columnsToNormalize);
elseif ischar(columnsToNormalize)
    columnsToNormalize = {columnsToNormalize};
end

normParams = struct();
datasNorm = datas;

% Loop through each column requested for normalization
for i = 1:numel(columnsToNormalize)
    colName = columnsToNormalize{i};
    vFull = datas.(colName);

    % Handle datetime arrays
    if isdatetime(vFull)
        mu = mean(vFull, "omitnat");
        sg = std(vFull, 0, "omitnat");

        % Prevent division by zero if all datetimes are identical
        sgIsZero = (seconds(sg) == 0) | isnan(seconds(sg));
        if sgIsZero
            sg = seconds(1);
        end

        % Handle duration arrays
    elseif isduration(vFull)
        mu = mean(vFull, "omitnan");
        sg = std(vFull, 0, "omitnan");

        % Prevent division by zero if all durations are identical
        sgIsZero = (seconds(sg) == 0) | isnan(seconds(sg));
        if sgIsZero
            sg = seconds(1);
        end

        % Handle numeric arrays (the standard case)
    else
        mu = mean(vFull, "omitnan");
        sg = std(vFull, 0, "omitnan");

        % Prevent division by zero if all numeric values are identical
        if isnan(sg) || sg == 0
            sg = 1;
        end
    end

    % Apply the z-score normalization formula: (x - mean) / std_dev
    zFull = (vFull - mu) ./ sg;
    datasNorm.(colName) = zFull;

    % Store the normalization parameters (mean and std) so they can be
    % retrieved later for de-normalization of predictions.
    safeField = matlab.lang.makeValidName(colName);
    normParams.(safeField).mean = mu;
    normParams.(safeField).std = sg;
end

fprintf("Z-score normalization completed for the entire dataset.\n");
end