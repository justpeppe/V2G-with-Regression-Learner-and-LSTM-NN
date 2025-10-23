function [XTrain, YTrain, time_vector_lags] = create_lags_LSTMNN(tbl, num_predittori, columns_table, target)
    
    % Verifica che esista la variabile time_vector e che sia datetime
    assert(ismember('time_vector', tbl.Properties.VariableNames), ...
        'La tabella deve contenere la variabile time_vector.');
    if ~isdatetime(tbl.time_vector)
        error('tbl.time_vector deve essere di tipo datetime.');
    end

    % Identifica i giorni unici
    giorni = dateshift(tbl.time_vector, 'start', 'day');
    giorni_unici = sort(unique(giorni));
    disp(['Giorni totali: ' num2str(height(giorni_unici))]);

    % Pre-calcola quali giorni sono validi (consecutivi) — info diagnostica
    giorni_consecutivi = false(height(giorni_unici), 1);
    for idx_giorni = 2 : height(giorni_unici)
        if days(giorni_unici(idx_giorni) - giorni_unici(idx_giorni - 1)) == 1
            giorni_consecutivi(idx_giorni) = true;
        end
    end
    giorni_validi = sum(giorni_consecutivi);
    disp(['Giorni consecutivi validi: ' num2str(giorni_validi)]);

    % Estrae le feature numeriche
    Xmat = table2array(tbl(:, columns_table));

    % Estrae i target
    if ischar(target) || (isstring(target) && isscalar(target))
        Ymat = table2array(tbl(:, target));
    else
        Ymat = table2array(tbl(:, target));
    end

    nRows = size(tbl, 1);
    numFeatures = size(Xmat, 2);
    numResponses = size(Ymat, 2);

    disp(['Lunghezza sequenza (num_predittori): ' num2str(num_predittori)]);
    disp(['Numero di feature: ' num2str(numFeatures)]);
    disp(['Numero di target: ' num2str(numResponses)]);

    % Preallocazione temporanea
    XTrain_temp = cell(nRows, 1);                % Sequenze features x time
    YTrain_temp = zeros(nRows, numResponses);    % Target riga-per-sequenza
    time_temp   = NaT(nRows, 1);                 % Timestamp associato alla riga di Y (t_target)
    idx_sequenza = 1;

    % Costruzione finestre solo per giorni consecutivi
    for i = 1:(nRows - num_predittori)
        t_end = i + num_predittori - 1;          % Ultimo indice della finestra
        t_target = t_end + 1;                    % Indice del target (sequence-to-one)

        % Verifica che tutti i timesteps della sequenza + target siano nello stesso gruppo di giorni consecutivi
        giorni_sequenza = giorni(i:t_target);
        giorni_sequenza_unici = unique(giorni_sequenza);

        % Controlla che i giorni attraversati siano consecutivi
        valida = true;
        if numel(giorni_sequenza_unici) > 1
            for j = 2:numel(giorni_sequenza_unici)
                if days(giorni_sequenza_unici(j) - giorni_sequenza_unici(j-1)) ~= 1
                    valida = false;
                    break;
                end
            end
        end

        % Se la sequenza è valida, aggiungi finestra, target e timestamp del target
        if valida
            XTrain_temp{idx_sequenza} = Xmat(i:t_end, :)';   % [features x time]
            YTrain_temp(idx_sequenza, :) = Ymat(t_target, :);
            time_temp(idx_sequenza, 1) = tbl.time_vector(t_target); % datetime associato alla riga di Y
            idx_sequenza = idx_sequenza + 1;
        end
    end

    % Taglio agli elementi validi
    last = idx_sequenza - 1;
    XTrain = XTrain_temp(1:last);
    YTrain = YTrain_temp(1:last, :);
    time_vector_lags = time_temp(1:last, 1);

    disp(['Numero di sequenze create: ' num2str(last)]);
    disp('Creazione sequenze LSTM completata');
end
