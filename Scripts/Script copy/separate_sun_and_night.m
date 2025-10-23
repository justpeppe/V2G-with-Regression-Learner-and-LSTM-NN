function [tabella_giorno, tabella_notte] = separate_sun_and_night(tabella_modello, zone_id, root)
% separate_sun_and_night Separa giorno e notte in due tabelle
%
%   INPUT:
%   - data:     La tabella con i dati completi (table).
%   - zone_id:  ID numerico della zona da caricare (numero).
%
%   OUTPUT:
%   - tabella_giorno: Tabella contenente i dati del giorno (table).
%   - tabella_notte: Tabella contenente i dati della notte (table).
switch zone_id
    case 8
        zone_full_name = 'Zone_1016_Anagnina';
    case 9
        zone_full_name = 'Zone_214_Trieste';
    case 10
        zone_full_name = 'Zone_2004_Della Vittoria,Tomba di Nerone,Tor di Quinto2';
    case 11
        zone_full_name = 'Zone2002_Tor di Quinto6';
    otherwise
        error('ID zona non valido o non gestito.'); % Aggiunge un controllo per ID non validi
end

% Caricamento dati alba e tramonto
    sun_datas_file = zone_full_name + "_Sun_datas.xlsx";
    sunrise_sunset = readtable(fullfile(root, "Sunrise and Sunset datas", sun_datas_file));
    sunrise_sunset.giorno = dateshift(sunrise_sunset.Sunrise, 'start', 'day');

    % Inizializzazione tabelle di output (mantenendo la struttura)
    tabella_giorno = tabella_modello(false, :);
    tabella_notte = tabella_modello(false, :);

    % Estrazione dei giorni unici dalla tabella di input
    giorni_tabella = dateshift(tabella_modello.time_vector, 'start', 'day');
    giorni_unici = unique(giorni_tabella);

    % Ciclo su ogni giorno per dividere i dati
    for idx_giorno = 1:height(giorni_unici)
        giorno_corrente = giorni_unici(idx_giorno);
        idx_sun = find(sunrise_sunset.giorno == giorno_corrente);

        if ~isempty(idx_sun)
            % Prendi solo il primo risultato se ce ne sono multipli per lo stesso giorno
            sunrise_time = sunrise_sunset.Sunrise(idx_sun(1));
            sunset_time = sunrise_sunset.Sunset(idx_sun(1));

            % Filtra i dati del giorno corrente dalla tabella modello
            dati_giorno_corrente = tabella_modello(giorni_tabella == giorno_corrente, :);

            % Identifica le righe diurne e notturne
            is_day_time = dati_giorno_corrente.time_vector >= sunrise_time & dati_giorno_corrente.time_vector < sunset_time;

            % Accoda i risultati alle tabelle di output (operazione lenta)
            tabella_giorno = [tabella_giorno; dati_giorno_corrente(is_day_time, :)];
            tabella_notte = [tabella_notte; dati_giorno_corrente(~is_day_time, :)];
        end
    end
end
