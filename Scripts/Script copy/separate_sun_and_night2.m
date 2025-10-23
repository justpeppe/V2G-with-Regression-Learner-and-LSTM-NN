function [tabella_giorno, tabella_notte] = separate_sun_and_night2(tabella_modello, orario_inizio, orario_fine)
% SEPARATE_SUN_AND_NIGHT2 Separa i dati in due tabelle in base a un intervallo orario.
%
%   Questa funzione prende una tabella contenente una colonna di datetime e la
%   divide in due tabelle: una per il "giorno" (l'intervallo specificato) e
%   una per la "notte" (tutti gli altri orari).
%
%   INPUT:
%   - tabella_modello: La tabella di input (table). Deve contenere una
%     colonna chiamata 'time_vector' di tipo datetime.
%   - orario_inizio: Stringa che rappresenta l'orario di inizio del giorno
%     (formato 'HH:mm:ss').
%   - orario_fine: Stringa che rappresenta l'orario di fine del giorno
%     (formato 'HH:mm:ss').
%
%   OUTPUT:
%   - tabella_giorno: Tabella con le righe che rientrano nell'intervallo diurno.
%   - tabella_notte: Tabella con le righe che non rientrano nell'intervallo diurno.

    % Assicura che la colonna 'time_vector' esista nella tabella
    if ~ismember('time_vector', tabella_modello.Properties.VariableNames)
        error("La tabella di input deve contenere una colonna chiamata 'time_vector'.");
    end

    % --- FIX APPLICATO QUI ---
    % Converte le stringhe di orario in oggetti datetime specificando il formato,
    % poi estrae un oggetto 'duration' usando timeofday.
    try
        inizio_dt = datetime(orario_inizio, 'InputFormat', 'HH:mm:ss');
        fine_dt = datetime(orario_fine, 'InputFormat', 'HH:mm:ss');
    catch ME
        error('Formato orario non valido. Assicurarsi che orario_inizio e orario_fine siano stringhe nel formato ''HH:mm:ss''.');
    end
    
    inizio_giorno = timeofday(inizio_dt);
    fine_giorno = timeofday(fine_dt);
    
    % Estrae la parte oraria dalla colonna 'time_vector' della tabella.
    orari_tabella = timeofday(tabella_modello.time_vector);
    
    % Crea la maschera logica per identificare le righe del "giorno".
    % Questa logica gestisce sia il caso normale sia quello con attraversamento della mezzanotte.
    if inizio_giorno < fine_giorno
        % Caso standard (es. da 06:00 a 20:00)
        idx_giorno = (orari_tabella >= inizio_giorno) & (orari_tabella <= fine_giorno);
    else
        % Caso in cui l'intervallo attraversa la mezzanotte (es. da 22:00 a 05:00)
        idx_giorno = (orari_tabella >= inizio_giorno) | (orari_tabella <= fine_giorno);
    end
    
    % Usa la maschera booleana per separare la tabella originale nelle due tabelle di output
    tabella_giorno = tabella_modello(idx_giorno, :);
    tabella_notte = tabella_modello(~idx_giorno, :);

end
