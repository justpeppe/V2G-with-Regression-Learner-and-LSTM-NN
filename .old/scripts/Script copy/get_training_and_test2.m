function [training, test] = get_training_and_test2(tabella_modello, campioni_per_giorno)
    % Inizializzazione tabelle di output vuote, mantenendo la struttura
    training = tabella_modello(false, :);
    test = tabella_modello(false, :);

    % Numero totale di righe nella tabella
    num_righe_totali = height(tabella_modello);
    
    start_idx = 1; % L'indicizzazione in MATLAB parte da 1

    giorno_lavorativo_da_scartare = 4;
    giorno_festivo_da_scartare = 6;
    
    % Continua a ciclare finché non abbiamo processato tutte le righe
    while start_idx <= num_righe_totali
        % Calcola l'indice di fine per la settimana corrente
        end_idx = start_idx + (campioni_per_giorno * 7) - 1;

        % Se l'indice di fine calcolato supera il numero di righe,
        % impostalo all'ultima riga disponibile.
        if end_idx > num_righe_totali
            end_idx = num_righe_totali;
        end

        % Estrai il blocco di dati per la settimana (o porzione di settimana)
        tabella_ridotta = tabella_modello(start_idx:end_idx, :);

        % Identifica le righe da inserire nel set di test
        idx_test_corrente = ismember(tabella_ridotta.giorno_settimana, [giorno_lavorativo_da_scartare, giorno_festivo_da_scartare]);

        % Accoda (concatena) le nuove righe alle tabelle esistenti
        test = [test; tabella_ridotta(idx_test_corrente, :)];
        training = [training; tabella_ridotta(~idx_test_corrente, :)];
        
        % Aggiorna l'indice di partenza per il ciclo successivo
        start_idx = end_idx + 1;

        if giorno_festivo_da_scartare == 6
            giorno_festivo_da_scartare = 7;
        else
            giorno_festivo_da_scartare = 6;
        end

        if giorno_lavorativo_da_scartare == 5
            giorno_lavorativo_da_scartare = 1;
        else
            giorno_lavorativo_da_scartare = giorno_lavorativo_da_scartare + 1;
        end
    end
end
