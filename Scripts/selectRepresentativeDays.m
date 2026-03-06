function [valDays, testDays] = selectRepresentativeDays(datas)
% selectRepresentativeDays Automatically selects representative validation
% and test day-pairs by detecting non-consecutive data clusters.
%
%   ALGORITHM
%   ---------
%   1. Identify distinct data clusters (gaps > 7 days).
%   2. From each cluster, select:
%        a. Best NORMAL day  : non-Sunday, closest to cluster median.
%        b. Best SUNDAY      : Sunday, closest to cluster Sunday median.
%   3. Distribute pairs to validation and test sets.
%
%   Usage: [valDays, testDays] = selectRepresentativeDays(datas);

arguments
    datas table
end

% ── 1. Daily Summary ─────────────────────────────────────────────────────
dayStamps  = dateshift(datas.time_vector, "start", "day");
uniqueDays = unique(dayStamps);
nDays      = numel(uniqueDays);

% Preallocate arrays for daily metrics
dailyTotal  = zeros(nDays, 1);
dailyTemp   = zeros(nDays, 1);
dailyPrecip = zeros(nDays, 1);
dailyWind   = zeros(nDays, 1);
dailyDow    = zeros(nDays, 1);

for k = 1:nDays
    mask = (dayStamps == uniqueDays(k));

    % Sum for energy, average for weather
    dailyTotal(k)  = sum(datas.AAC_energy(mask), "omitnan");
    dailyTemp(k)   = mean(datas.temp(mask), "omitnan");
    dailyPrecip(k) = mean(datas.precipprob(mask), "omitnan");
    dailyWind(k)   = mean(datas.windspeed(mask), "omitnan");

    dailyDow(k)    = day(uniqueDays(k), "dayofweek");
end

% ── 2. Cluster Detection (Gap > 7 days) ──────────────────────────────────
gaps     = days(diff(uniqueDays)) > 7;
clusterStartIdx = [1; find(gaps) + 1];
clusterEndIdx   = [find(gaps); nDays];
numClusters     = numel(clusterStartIdx);

fprintf("Detected %d data clusters:\n", numClusters);
for c = 1:numClusters
    fprintf("  Cluster %d: [%s] to [%s]\n", c, ...
        string(uniqueDays(clusterStartIdx(c)), "yyyy-MM-dd"), ...
        string(uniqueDays(clusterEndIdx(c)), "yyyy-MM-dd"));
end

clusterPairs = {};

% ── 3. Extract Normal + Sunday from each cluster ─────────────────────────
for c = 1:numClusters
    idx = clusterStartIdx(c):clusterEndIdx(c);
    cDays   = uniqueDays(idx);
    cTots   = dailyTotal(idx);
    cTemps  = dailyTemp(idx);
    cPrecip = dailyPrecip(idx);
    cWind   = dailyWind(idx);
    cDow    = dailyDow(idx);

    % Find Normal Day (non-Sunday)
    normMask = (cDow ~= 1);
    if any(normMask)
        [normDay, normReg] = pickBestDay(cDays(normMask), cTots(normMask), cTemps(normMask), cPrecip(normMask), cWind(normMask), uniqueDays);
        if ~isempty(normDay)
            clusterPairs{end+1} = [normReg; normDay]; %#ok<AGROW>
        end
    end

    % Find Sunday
    sunMask = (cDow == 1);
    if any(sunMask)
        [sunDay, sunReg] = pickBestDay(cDays(sunMask), cTots(sunMask), cTemps(sunMask), cPrecip(sunMask), cWind(sunMask), uniqueDays);
        if ~isempty(sunDay)
            clusterPairs{end+1} = [sunReg; sunDay]; %#ok<AGROW>
        end
    end
end

% ── 4. Assign to Val/Test ────────────────────────────────────────────────
% We need 4 pairs total (2 for val, 2 for test).
if numel(clusterPairs) < 4
    error("Not enough representative pairs found (%d/4). Check data gaps.", numel(clusterPairs));
end

% Pick 4 pairs (largest clusters if many)
finalPairs = clusterPairs(1:min(4, end));

valDays  = vertcat(finalPairs{1}, finalPairs{3});
testDays = vertcat(finalPairs{2}, finalPairs{4});

% ── 5. Summary ───────────────────────────────────────────────────────────
fprintf("\n--- Dynamic Cluster Selection ---\n");
fprintf("VALIDATION:\n  [%s] Normal | [%s] Sunday\n", string(valDays(2), "yyyy-MM-dd"), string(valDays(4), "yyyy-MM-dd"));
fprintf("TEST:\n  [%s] Sunday | [%s] Normal\n", string(testDays(2), "yyyy-MM-dd"), string(testDays(4), "yyyy-MM-dd"));
fprintf("---------------------------------\n\n");

end

function [selected, regressor] = pickBestDay(days, totals, temps, precips, winds, allUniqueDays)
selected = []; regressor = [];

% 1. Primary Filter: Energy Distance
% The day MUST be representative of normal consumption.
medEnergy = median(totals, "omitnan");
energyDist = abs(totals - medEnergy);

% Sort days purely by their energy typicality
[~, ordEnergy] = sort(energyDist);

% Select the top 5 closest days as our candidate pool (or fewer if cluster is small)
poolSize = min(5, numel(ordEnergy));
poolIdx = ordEnergy(1:poolSize);

% 2. Secondary Filter: Weather Distance (The Tie-Breaker)
% Among the top energy candidates, find the one with the most normal weather.
weatherFeatures = [temps(poolIdx), precips(poolIdx), winds(poolIdx)];

if size(weatherFeatures, 1) > 1
    normWeather = normalize(weatherFeatures, 1, "zscore");
else
    normWeather = weatherFeatures;
end

% Compute median weather profile FOR THE ENTIRE CLUSTER (not just the pool)
% to see how our pool candidates compare to the cluster norm
clusterWeather = [temps, precips, winds];
if size(clusterWeather, 1) > 1
    normClusterWeather = normalize(clusterWeather, 1, "zscore");
else
    normClusterWeather = clusterWeather;
end
medWeather = median(normClusterWeather, 1, "omitnan");

% Calculate distance of our pooled candidates to the cluster weather median
weatherDist = sqrt(sum((normWeather - medWeather).^2, 2));

% Sort our specific pool by weather distance
[~, ordWeather] = sort(weatherDist);
finalOrdIdx = poolIdx(ordWeather);

% Iterate through the final candidates until we find one with a valid historical regressor (T-1)
for i = 1:numel(finalOrdIdx)
    candidate = days(finalOrdIdx(i));
    regDay = candidate - caldays(1);
    if ismember(regDay, allUniqueDays)
        selected = candidate;
        regressor = regDay;
        break;
    end
end
end

