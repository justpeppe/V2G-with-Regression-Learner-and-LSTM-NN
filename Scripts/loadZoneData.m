function dataOut = loadZoneData(projectRoot, zoneId)
% loadZoneData Loads data for a specific zone and merges it with weather data.
arguments
    projectRoot (1,1) string
    zoneId (1,1) double {mustBeInteger, mustBePositive}
end


%% Load zone data based on ID
% Dynamically select the correct MAT file and variable name based on the requested zone ID.
switch zoneId
    case 8
        zoneFullName = "Zone_1016_Anagnina";
        zoneData = load(fullfile(projectRoot, "Dati Estratti (from Condivisione)", "Zone1016_new.mat"), "AAC_energy", "time_vector");
    case 9
        zoneFullName = "Zone_214_Trieste";
        zoneData = load(fullfile(projectRoot, "Dati Estratti (from Condivisione)", "Zone214_new.mat"), "AAC_energy", "time_vector");
    case 10
        zoneFullName = "Zone_2004_Della Vittoria,Tomba di Nerone,Tor di Quinto2";
        zoneData = load(fullfile(projectRoot, "Dati Estratti (from Condivisione)", "Zone2004_new.mat"), "AAC_energy", "time_vector");
    case 11
        zoneFullName = "Zone2002_Tor di Quinto6";
        zoneData = load(fullfile(projectRoot, "Dati Estratti (from Condivisione)", "Zone2002_new.mat"), "AAC_energy", "time_vector");
    case {1, 2, 3, 4, 5, 6, 7, 12}
        error("Zone ID %d is theoretically defined but load file is not mapped.", zoneId);
    otherwise
        error("Zone ID not valid or not handled. Supported values: 8, 9, 10, 11.");
end

%% Load weather and holiday data
% Load exogenous variables which are shared across all zones
meteoData = load(fullfile(projectRoot, "Gabriele Datas", "metero_year_hh.mat"), "meteo_year_hh");
holidaysData = load(fullfile(projectRoot, "Gabriele Datas", "holidays.mat"), "vacanze");

%% Final dataset creation
% Merge the zone-specific target data with the overarching weather and holiday datasets.
% The meteorological table acts as the base table to which we append new columns.
dataOut = meteoData.meteo_year_hh;
dataOut.holiday_indicator = holidaysData.vacanze;
dataOut.AAC_energy = zoneData.AAC_energy{:, zoneFullName};
dataOut.time_vector = zoneData.time_vector(:);

fprintf("Data for zone ""%s"" loaded correctly.\n", zoneFullName);

end
