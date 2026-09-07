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

% Columns are appended by position. A zone file with the same number of rows but a
% different time base would misalign every row without raising an error, so compare the
% two time axes here, while they are still separate.
if ~isequal(data.datetime, zoneData.time_vector(:))
    error("loadZoneData:timeMismatch", ...
        "Zone %d time base differs from the weather data; the positional merge would be wrong.", zoneId);
end

% Not a 0/1 flag despite the old name: an exponential ramp of proximity to the next
% non-working day. Working days peak at 0.9583, non-working days reach 1 within
% floating point error, so classify with a > 0.99 threshold and never with == 1.
data.holiday_proximity = holidaysData.vacanze;
data.AAC_energy = zoneData.AAC_energy{:, zoneFullName};

% Single time column, in ISO 8601: it sorts chronologically even when read as plain text
% and needs no locale, unlike dd-MMM-yyyy. The zone time_vector is dropped because the
% check above proved it identical.
data.datetime.Format = 'yyyy-MM-dd''T''HH:mm:ss';

fprintf("Data for zone ""%s"" loaded correctly (%d rows x %d columns).\n", ...
    zoneFullName, height(data), width(data));

end