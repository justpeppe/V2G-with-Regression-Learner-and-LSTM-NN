function data_out = load_datas(root, zone_id)
% load_datas Carica i dati per una zona specifica e li unisce ai dati meteo.
%
%   INPUT:
%   - root:     Percorso della cartella principale del progetto (stringa).
%   - zone_id:  ID numerico della zona da caricare (numero).
%
%   OUTPUT:
%   - data_out: Tabella contenente i dati combinati (table).

%% Caricamento dati zona in base all'ID
switch zone_id
    case 8
        zone_full_name = 'Zone_1016_Anagnina';
        load(fullfile(root, "Dati Estratti (from Condivisione)", "Zone1016_new.mat"));
    case 9
        zone_full_name = 'Zone_214_Trieste';
        load(fullfile(root, "Dati Estratti (from Condivisione)", "Zone214_new.mat"));
    case 10
        zone_full_name = 'Zone_2004_Della Vittoria,Tomba di Nerone,Tor di Quinto2';
        load(fullfile(root, "Dati Estratti (from Condivisione)", "Zone2004_new.mat"));
    case 11
        zone_full_name = 'Zone2002_Tor di Quinto6';
        load(fullfile(root, "Dati Estratti (from Condivisione)", "Zone2002_new.mat"));
    otherwise
        error('ID zona non valido o non gestito.'); % Aggiunge un controllo per ID non validi
end

%% Caricamento dati meteo e holidays
load(fullfile(root, "Gabriele Datas", "metero_year_hh.mat"));
load(fullfile(root, "Gabriele Datas", "holidays.mat"));

%% Creazione del dataset finale
data_out = meteo_year_hh;
data_out.holiday_indicator = vacanze;
data_out.AAC_energy = AAC_energy{:, zone_full_name};
data_out.time_vector = time_vector(:);

disp(['Dati per la zona "', zone_full_name, '" caricati correttamente.']);

end
