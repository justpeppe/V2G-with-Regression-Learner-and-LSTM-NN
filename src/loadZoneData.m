function data = loadZoneData(dataFolder, zoneId)
% loadZoneData Loads data for a specific zone and merges it with weather data.
arguments
    dataFolder (1,1) string
    zoneId (1,1) double {mustBeInteger, mustBePositive}
end


%% Load zone data based on ID
% Dynamically select the correct MAT file and variable name based on the requested zone ID.
switch zoneId
    case 8
        zoneFullName = "Zone_1016_Anagnina";
        zoneData = load(fullfile(dataFolder, "Dati Estratti (from Condivisione)", "Zone1016_new.mat"), "AAC_energy", "time_vector");
    case 9
        zoneFullName = "Zone_214_Trieste";
        zoneData = load(fullfile(dataFolder, "Dati Estratti (from Condivisione)", "Zone214_new.mat"), "AAC_energy", "time_vector");
    case 10
        zoneFullName = "Zone_2004_Della Vittoria,Tomba di Nerone,Tor di Quinto2";
        zoneData = load(fullfile(dataFolder, "Dati Estratti (from Condivisione)", "Zone2004_new.mat"), "AAC_energy", "time_vector");
    case 11
        zoneFullName = "Zone2002_Tor di Quinto6";
        zoneData = load(fullfile(dataFolder, "Dati Estratti (from Condivisione)", "Zone2002_new.mat"), "AAC_energy", "time_vector");
    case {1, 2, 3, 4, 5, 6, 7, 12}
        error("Zone ID %d is theoretically defined but load file is not mapped.", zoneId);
    otherwise
        error("Zone ID not valid or not handled. Supported values: 8, 9, 10, 11.");
end

%% Load weather and holiday data
% Load exogenous variables which are shared across all zones
meteoData = load(fullfile(dataFolder, "Gabriele Datas", "metero_year_hh.mat"), "meteo_year_hh");
holidaysData = load(fullfile(dataFolder, "Gabriele Datas", "holidays.mat"), "vacanze");

%% Final dataset creation
% Merge the zone-specific target data with the overarching weather and holiday datasets.
% The meteorological table acts as the base table to which we append new columns.
data = meteoData.meteo_year_hh;

% Compare time axes while they are still separate
if ~isequal(data.datetime, zoneData.time_vector(:))
    error("loadZoneData:timeMismatch", ...
        "Zone %d time base differs from the weather data; the positional merge would be wrong.", zoneId);
end

% Not a 0/1 flag: an exponential ramp of proximity to the next non-working day
data.holiday_proximity = holidaysData.vacanze;
data.AAC_energy = zoneData.AAC_energy{:, zoneFullName};

% Single time column in ISO 8601
data.datetime.Format = 'yyyy-MM-dd''T''HH:mm:ss';

% Verify target integrity
if any(isnan(data.AAC_energy))
    error("loadZoneData:missingTarget", "Zone %d contains NaN values in target energy.", zoneId);
end

%% Calendar properties and cluster logging
% Extract calendar day properties and seasonal cluster boundaries
daysVec = dateshift(data.datetime, "start", "day");
uniqueDays = unique(daysVec);
numDays = numel(uniqueDays);

isNonWorkingDay = false(size(uniqueDays));
for i = 1:numDays
    dayMask = daysVec == uniqueDays(i);
    isHoliday = max(data.holiday_proximity(dayMask)) > 0.99;
    isNonWorkingDay(i) = isweekend(uniqueDays(i)) | isHoliday;
end

numWorking = sum(~isNonWorkingDay);
numNonWorking = sum(isNonWorkingDay);

isNonWorkingRow = isweekend(data.datetime) | (data.holiday_proximity > 0.99);
meanWorkEnergy = mean(data.AAC_energy(~isNonWorkingRow));
meanNonWorkEnergy = mean(data.AAC_energy(isNonWorkingRow));

dayGaps = days(diff(uniqueDays));
clusterStarts = [1; find(dayGaps > 1) + 1];
clusterEnds = [find(dayGaps > 1); numDays];
numClusters = numel(clusterStarts);

% Store structured metadata in table properties
data.Properties.UserData.zoneName = zoneFullName;
data.Properties.UserData.uniqueDays = uniqueDays;
data.Properties.UserData.isNonWorkingDay = isNonWorkingDay;
data.Properties.UserData.clusterStarts = clusterStarts;
data.Properties.UserData.clusterEnds = clusterEnds;

fprintf("Zone ""%s"" loaded: %d rows (%d complete days, 48 steps/day).\n", ...
    zoneFullName, height(data), numDays);
fprintf("Day breakdown: %d working days (mean: %.1f kWh), %d non-working days (mean: %.1f kWh).\n", ...
    numWorking, meanWorkEnergy, numNonWorking, meanNonWorkEnergy);
fprintf("Seasonal clusters identified (%d clusters):\n", numClusters);
for c = 1:numClusters
    sStr = string(uniqueDays(clusterStarts(c)), "yyyy-MM-dd (eee)");
    eStr = string(uniqueDays(clusterEnds(c)), "yyyy-MM-dd (eee)");
    nDays = clusterEnds(c) - clusterStarts(c) + 1;
    fprintf("  Cluster %d: %s to %s (%d days)\n", c, sStr, eStr, nDays);
end

end