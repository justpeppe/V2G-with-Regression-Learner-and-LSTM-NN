function [training_norm, test_norm, parametri_norm] = normalize_zscore(training_set, test_set, colonne_da_normalizzare)
    % normalize_zscore: normalizza colonne specifiche (numeric, datetime, duration)
    % usando z-score con parametri calcolati sul solo training.
    %
    % Output:
    %  - training_norm/test_norm: ONLY le colonne normalizzate (più time_vector se presente nella tabella)
    %  - parametri_norm: struct con media e dev_std (tipi coerenti: double/datetime/duration)

    % 1) Rileva tipo input (tabella o matrice)
    is_table_input = istable(training_set);                    % true se tabella

    % 2) Coerenza tipi tra training e test
    assert(istable(test_set) == is_table_input, ...
        'Training e test devono essere entrambi tabella o entrambi matrice.'); % evita mismatch

    % 3) Normalizza "colonne_da_normalizzare" in formato atteso
    if is_table_input
        % Accetta string array, cellstr, char
        if isstring(colonne_da_normalizzare)
            colonne_da_normalizzare = cellstr(colonne_da_normalizzare);        % converte string in cellstr
        elseif ischar(colonne_da_normalizzare)
            colonne_da_normalizzare = {colonne_da_normalizzare};               % singolo nome char in cell
        end
        assert(iscellstr(colonne_da_normalizzare), ...
            'Per tabelle, colonne_da_normalizzare deve essere cell array di nomi di variabile.'); % validazione
        % Verifica esistenza colonne
        assert(all(ismember(colonne_da_normalizzare, training_set.Properties.VariableNames)), ...
            'Alcune colonne non esistono nel training_set.');                  % sicurezza
        assert(all(ismember(colonne_da_normalizzare, test_set.Properties.VariableNames)), ...
            'Alcune colonne non esistono nel test_set.');                      % sicurezza
    else
        % Per matrici: ci aspettiamo indici numerici
        assert(isnumeric(colonne_da_normalizzare), ...
            'Per matrici, colonne_da_normalizzare deve essere vettore di indici numerici.'); % validazione
        assert(all(colonne_da_normalizzare >= 1 & colonne_da_normalizzare <= size(training_set,2)), ...
            'Indici fuori dal range del training_set.');                        % bound check
        assert(all(colonne_da_normalizzare >= 1 & colonne_da_normalizzare <= size(test_set,2)), ...
            'Indici fuori dal range del test_set.');                            % bound check
    end

    % 4) Prepara output e struct parametri
    parametri_norm = struct();                                                  % init parametri
    if is_table_input
        training_norm = table();                                                % tabella vuota (solo col norm)
        test_norm = table();                                                    % tabella vuota (solo col norm)
    else
        num_cols = numel(colonne_da_normalizzare);                              % n colonne da normalizzare
        training_norm = zeros(size(training_set,1), num_cols);                  % prealloc matrice training
        test_norm = zeros(size(test_set,1), num_cols);                          % prealloc matrice test
    end

    % 5) Loop colonne da normalizzare
    for i = 1:numel(colonne_da_normalizzare)
        if is_table_input
            col_nome = colonne_da_normalizzare{i};                              % nome colonna
            v_train = training_set.(col_nome);                                  % dati training colonna originale
            v_test  = test_set.(col_nome);                                      % dati test colonna originale
            % Tipi ammessi: numeric, datetime, duration
            if ~(isnumeric(v_train) || isdatetime(v_train) || isduration(v_train))
                error('La colonna %s non è numeric/datetime/duration: impossibile normalizzare.', col_nome);
            end
        else
            col_idx = colonne_da_normalizzare(i);                               % indice colonna (matrice)
            v_train = training_set(:, col_idx);                                 % vettore training (double)
            v_test  = test_set(:, col_idx);                                     % vettore test (double)
        end

        % 6) Calcolo media e dev std sul SOLO training, ignorando NaN
        if isdatetime(v_train)
            mu = mean(v_train, 'omitnat');                                      % media datetime (NaT ignorati)
            sg = std(v_train, 0, 'omitnat');                                    % std duration
            sg_is_zero = (seconds(sg) == 0) | isnan(seconds(sg));               % zero/NaN-safe su duration
            if sg_is_zero, sg = seconds(1); end                                 % evita divisione per zero
            z_train = (v_train - mu) ./ sg;                                     % duration/duration -> double
            z_test  = (v_test  - mu) ./ sg;                                     % normalizza test con mu, sg del training
        elseif isduration(v_train)
            mu = mean(v_train, 'omitnan');                                      % media duration
            sg = std(v_train, 0, 'omitnan');                                    % std duration
            sg_is_zero = (seconds(sg) == 0) | isnan(seconds(sg));               % controlla zero/NaN
            if sg_is_zero, sg = seconds(1); end                                 % fallback
            z_train = (v_train - mu) ./ sg;                                     % duration/duration -> double
            z_test  = (v_test  - mu) ./ sg;                                     % normalizza test
        else
            mu = mean(v_train, 'omitnan');                                      % media numerica
            sg = std(v_train, 0, 'omitnan');                                    % std numerica
            if isnan(sg) || sg == 0, sg = 1; end                                % evita divisione per zero/NaN
            z_train = (v_train - mu) ./ sg;                                     % z-score training
            z_test  = (v_test  - mu) ./ sg;                                     % z-score test
        end

        % 7) Scrive output normalizzato
        if is_table_input
            training_norm.(col_nome) = z_train;                                 % aggiunge colonna normalizzata
            test_norm.(col_nome) = z_test;                                      % aggiunge colonna normalizzata
            safe_field = matlab.lang.makeValidName(col_nome);                   % nome campo valido per struct
            parametri_norm.(safe_field).media  = mu;                            % salva media (tipo coerente)
            parametri_norm.(safe_field).dev_std = sg;                           % salva std (double o duration)
        else
            training_norm(:, i) = z_train;                                      % inserisce colonna i
            test_norm(:, i) = z_test;                                           % inserisce colonna i
            pname = sprintf('colonna_%d', col_idx);                             % nome campo per struct
            parametri_norm.(pname).media  = mu;                                  % salva media numerica
            parametri_norm.(pname).dev_std = sg;                                 % salva std numerica
        end
    end

    % 8) Aggiunge time_vector solo per tabelle e solo se esiste nell'input
    if is_table_input && ismember('time_vector', training_set.Properties.VariableNames) ...
                      && ismember('time_vector', test_set.Properties.VariableNames)
        training_norm.time_vector = training_set.time_vector;                   % copia time_vector (non normalizzato)
        test_norm.time_vector = test_set.time_vector;                           % copia time_vector (non normalizzato)
    end

    disp('Normalizzate tabelle con metodo z-score');
end
