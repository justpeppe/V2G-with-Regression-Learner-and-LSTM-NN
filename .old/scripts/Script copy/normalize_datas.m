function tabella_normalizzata = normalize_datas(tabella_input)
% normalize_datas Normalizza tutte le colonne numeriche di una tabella.
%
% REGOLE:
% - La colonna 'datetime' viene rimossa.
% - La colonna 'time_vector' (di tipo datetime) viene convertita in numerico e poi normalizzata.
% - Tutte le altre colonne numeriche vengono normalizzate nell'intervallo [0, 1].
% - Le colonne non numeriche (es. stringhe) vengono mantenute invariate.
%
% INPUT:
%   - tabella_input: La tabella di origine da normalizzare.
%
% OUTPUT:
%   - tabella_normalizzata: La nuova tabella con i dati normalizzati.

    % Crea una copia per non modificare la tabella originale
    tabella_normalizzata = tabella_input;
    
    % 1. Gestione colonna 'datetime'
    %    Controlla se la colonna 'datetime' esiste e, in caso affermativo, la rimuove.
    if ismember('datetime', tabella_normalizzata.Properties.VariableNames)
        tabella_normalizzata.datetime = [];
        disp("Colonna 'datetime' rimossa.");
    end
    
    % 2. Gestione colonna 'time_vector'
    %    Controlla se la colonna 'time_vector' esiste e se è di tipo datetime.
    if ismember('time_vector', tabella_normalizzata.Properties.VariableNames) && isdatetime(tabella_normalizzata.time_vector)
        % Converte il formato datetime in un numero (secondi POSIX/Unix time)
        tabella_normalizzata.time_vector = posixtime(tabella_normalizzata.time_vector);
        disp("Colonna 'time_vector' convertita da datetime a numerico (posixtime).");
    end
    
    % 3. Normalizzazione selettiva
    %    Applica la normalizzazione Min-Max (range [0, 1]) SOLO alle colonne
    %    di tipo numerico, lasciando inalterate le altre.
    
    % Identifica quali colonne sono numeriche
    colonne_numeriche_idx = varfun(@isnumeric, tabella_normalizzata, 'OutputFormat', 'uniform');
    
    % Se sono state trovate colonne numeriche, procedi con la normalizzazione
    if any(colonne_numeriche_idx)
        % Estrai solo i dati numerici in una tabella temporanea
        dati_numerici = tabella_normalizzata(:, colonne_numeriche_idx);
        
        % Normalizza la tabella contenente solo dati numerici
        dati_normalizzati = normalize(dati_numerici, 'range');
        
        % Sostituisci le colonne numeriche della tabella originale con quelle normalizzate
        tabella_normalizzata(:, colonne_numeriche_idx) = dati_normalizzati;
        
        disp('Normalizzazione completata sulle colonne numeriche.');
    else
        disp('Nessuna colonna numerica trovata da normalizzare.');
    end
end
