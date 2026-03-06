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

for i = 1:numel(columnsToNormalize)
    colName = columnsToNormalize{i};
    vFull = datas.(colName);

    if isdatetime(vFull)
        mu = mean(vFull, "omitnat");
        sg = std(vFull, 0, "omitnat");
        sgIsZero = (seconds(sg) == 0) | isnan(seconds(sg));
        if sgIsZero
            sg = seconds(1);
        end
    elseif isduration(vFull)
        mu = mean(vFull, "omitnan");
        sg = std(vFull, 0, "omitnan");
        sgIsZero = (seconds(sg) == 0) | isnan(seconds(sg));
        if sgIsZero
            sg = seconds(1);
        end
    else
        mu = mean(vFull, "omitnan");
        sg = std(vFull, 0, "omitnan");
        if isnan(sg) || sg == 0
            sg = 1;
        end
    end

    zFull = (vFull - mu) ./ sg;
    datasNorm.(colName) = zFull;

    safeField = matlab.lang.makeValidName(colName);
    normParams.(safeField).mean = mu;
    normParams.(safeField).std = sg;
end

fprintf("Z-score normalization completed for the entire dataset.\n");
end