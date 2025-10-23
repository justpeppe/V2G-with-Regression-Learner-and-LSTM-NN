function [training_norm, test_norm, parametri_norm] = normalize_zscore(training_set, test_set, colonne_da_normalizzare)
% normalize_zscore_datetime: Normalizza colonne specifiche, gestendo le colonne datetime.
%
% Input:
%   training_set: Tabella o matrice di dati di training.
%   test_set: Tabella o matrice di dati di test.
%   colonne_da_normalizzare: Cell array di stringhe (per tabelle) o array numerico 
%                            di indici (per matrici) delle colonne da normalizzare.
%
% Output:
%   training_norm: Tabella o matrice con SOLO le colonne di training normalizzate.
%   test_norm: Tabella o matrice con SOLO le colonne di test normalizzate.
%   parametri_norm: Struct con media e deviazione standard usate per la normalizzazione.
%

parametri_norm = struct();
is_table_input = istable(training_set);

% Inizializza gli output come vuoti
if is_table_input
    training_norm = table();
    test_norm = table();
else
    % Pre-alloca le matrici per efficienza
    num_cols = length(colonne_da_normalizzare);
    training_norm = zeros(size(training_set, 1), num_cols);
    test_norm = zeros(size(test_set, 1), num_cols);
end

for i = 1:length(colonne_da_normalizzare)
    if is_table_input
        col_nome = colonne_da_normalizzare{i};
        
        % Estrai colonne di training e test
        col_training_originale = training_set.(col_nome);
        col_test_originale = test_set.(col_nome);

        % --- GESTIONE DATETIME ---
        % Se la colonna si chiama 'time_vector' ed è di tipo datetime, convertila
        if strcmp(col_nome, 'time_vector') && isdatetime(col_training_originale)
            % Converte datetime in numero (secondi POSIX/Unix time)
            dati_training_col = posixtime(col_training_originale);
            dati_test_col = posixtime(col_test_originale);
        else
            dati_training_col = col_training_originale;
            dati_test_col = col_test_originale;
        end
        % --- FINE GESTIONE DATETIME ---

    else % Caso Matrice (non può contenere datetime)
        col_idx = colonne_da_normalizzare(i);
        dati_training_col = training_set(:, col_idx);
        dati_test_col = test_set(:, col_idx);
    end
    
    % 1. Calcola media e deviazione standard SOLO dal training set (ora numerico)
    media = mean(dati_training_col);
    dev_std = std(dati_training_col);
    
    % Evita divisione per zero
    if dev_std == 0
        dev_std = 1;
    end
    
    % Salva i parametri
    if is_table_input
        param_nome = col_nome;
    else
        param_nome = sprintf('colonna_%d', col_idx);
    end
    parametri_norm.(param_nome).media = media;
    parametri_norm.(param_nome).dev_std = dev_std;
    
    % 2. Applica la normalizzazione a entrambi i set di dati
    dati_training_normalizzati = (dati_training_col - media) / dev_std;
    dati_test_normalizzati = (dati_test_col - media) / dev_std;
    
    % 3. Aggiungi la colonna normalizzata al nuovo dataset
    if is_table_input
        training_norm.(col_nome) = dati_training_normalizzati;
        test_norm.(col_nome) = dati_test_normalizzati;
    else
        training_norm(:, i) = dati_training_normalizzati;
        test_norm(:, i) = dati_test_normalizzati;
    end
end
end
