function [dati_con_regressori] = create_lags_regression_learner(dati_input, regressori, nomiPredittori, nomeTarget)
    
    tabella_input = dati_input;
    
    % Verifica che esista la variabile time_vector e che sia datetime
    assert(ismember('time_vector', tabella_input.Properties.VariableNames), ...
        'La tabella deve contenere la variabile time_vector.');
    if ~isdatetime(tabella_input.time_vector)
        error('tabella.time_vector deve essere di tipo datetime.');
    end
    
    % Identifica i giorni unici
    giorni = dateshift(tabella_input.time_vector, 'start', 'day');
    giorni_unici = sort(unique(giorni));
    disp('Giorni totali: ' + string(height(giorni_unici)));
    
    % Pre-calcola quali giorni sono validi (consecutivi)
    giorni_consecutivi = false(height(giorni_unici), 1);
    for idx_giorni = 2 : height(giorni_unici)
        if days(giorni_unici(idx_giorni) - giorni_unici(idx_giorni - 1)) == 1
            giorni_consecutivi(idx_giorni) = true;
        end
    end
    
    giorni_validi = sum(giorni_consecutivi);
    disp(['Giorni consecutivi validi: ' num2str(giorni_validi)]);
    
    % Pre-alloca le matrici per i dati
    num_predittori = numel(nomiPredittori);
    num_colonne = num_predittori * regressori + 1; % +1 per il target
    righe_stimate = giorni_validi * regressori;
    
    % Pre-alloca matrice numerica
    dati_matrix = NaN(righe_stimate, num_colonne);
    idx_riga_output = 1;
    
    % Genera i nomi delle colonne
    nomi_colonne = cell(1, num_colonne);
    col_idx = 1;
    for lag = regressori : -1 : 1
        for idx_pred = 1 : num_predittori
            nomi_colonne{col_idx} = sprintf('%s_t_%d', nomiPredittori{idx_pred}, lag);
            col_idx = col_idx + 1;
        end
    end
    nomi_colonne{end} = nomeTarget;
    
    % Processa solo i giorni consecutivi validi
    for idx_giorni = find(giorni_consecutivi)'
        giorno_corrente = giorni_unici(idx_giorni);
        giorno_precedente = giorni_unici(idx_giorni - 1);
        
        % Trova gli indici per entrambi i giorni
        idx_giorno_corrente = giorni == giorno_corrente;
        idx_giorno_precedente = giorni == giorno_precedente;
        
        % Estrai i dati dei due giorni
        dati_precedente = tabella_input(idx_giorno_precedente, :);
        dati_corrente = tabella_input(idx_giorno_corrente, :);
        
        % Concatena in ordine temporale
        tabella_concatenata = [dati_precedente; dati_corrente];
        num_campioni_totali = height(tabella_concatenata);
        
        % Processa ogni campione del giorno corrente
        for idx_campione = (regressori + 1) : num_campioni_totali
            
            col_idx = 1;
            % Cicla sui lag (da t-regressori a t-1)
            for lag = regressori : -1 : 1
                idx_lag = idx_campione - lag;
                
                % Per ogni predittore al lag corrente
                for idx_pred = 1 : num_predittori
                    nome_pred = nomiPredittori{idx_pred};
                    dati_matrix(idx_riga_output, col_idx) = tabella_concatenata.(nome_pred)(idx_lag);
                    col_idx = col_idx + 1;
                end
            end
            
            % Aggiungi il valore target al tempo t
            dati_matrix(idx_riga_output, end) = tabella_concatenata.(nomeTarget)(idx_campione);
            idx_riga_output = idx_riga_output + 1;
        end
    end
    
    % Rimuovi eventuali righe NaN non utilizzate
    dati_matrix = dati_matrix(1:idx_riga_output-1, :);
    
    % Converti in tabella
    dati_con_regressori = array2table(dati_matrix, 'VariableNames', nomi_colonne);
    
    disp(['Righe totali create: ' num2str(height(dati_con_regressori))]);
end
