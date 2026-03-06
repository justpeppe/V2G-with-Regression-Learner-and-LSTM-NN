function [XTrain, YTrain] = table_LSTMNN_v2(tbl, num_predittori, columns_table, target)
    % Estrae le feature numeriche dalle colonne richieste (n-by-numFeatures) [web:39][web:9]
    Xmat = table2array(tbl(:, columns_table));                  % matrice double di feature [web:39][web:9]

    % Estrae i target come matrice numerica n-by-numResponses [web:9][web:22]
    if ischar(target) || (isstring(target) && isscalar(target)) % singolo nome variabile [web:9][web:22]
        Ymat = tbl.(target);                                    % vettore/colonna target [web:9][web:22]
    elseif isnumeric(target)                                    % indici di colonna target [web:9][web:22]
        Ymat = table2array(tbl(:, target));                     % matrice target [web:9][web:22]
    elseif isstring(target) || iscellstr(target)                % lista di nomi target [web:9][web:22]
        Ymat = table2array(tbl(:, target));                     % matrice target [web:9][web:22]
    else
        error('target deve essere nome, vettore di nomi o indici di colonna'); % validazione [web:9][web:22]
    end

    % Parametri di sliding window [web:9][web:22]
    sequenceLength = num_predittori;                            % lunghezza finestra (es. 48) [web:9][web:22]
    nRows = size(Xmat, 1);                                      % numero di righe disponibili [web:39][web:9]

    % Numero di sequenze utilizzabili senza eccedere [web:9][web:22]
    numSequences = nRows - sequenceLength;                      % N = n - s [web:9][web:22]
    if numSequences <= 0
        error('Tabella troppo corta per la finestra specificata'); % guardia [web:9][web:22]
    end

    % Prealloc: X come cell array N-by-1 di sequenze s-by-c (timesteps-by-features) [web:39][web:9]
    XTrain = cell(numSequences, 1);                             % cell array per trainnet [web:39][web:9]

    % Prealloc: Y come matrice N-by-R (sequence-to-one) [web:39][web:9]
    numResponses = size(Ymat, 2);                               % R variabili target [web:9][web:22]
    YTrain = zeros(numSequences, numResponses, 'like', Xmat);   % double N-by-R [web:39][web:9]

    % Costruzione finestre e target a +1 passo [web:9][web:22]
    for i = 1:numSequences                                      % itera su tutte le finestre [web:39][web:9]
        t_end = i + sequenceLength - 1;                         % indice finale finestra [web:9][web:22]
        XTrain{i} = Xmat(i:t_end, :)';                           % sequenza s-by-c (s=sequenceLength) [web:39][web:9]
        YTrain(i, :) = Ymat(t_end + 1, :);                      % riga successiva come target [web:9][web:22]
    end

    % (Opzionale) Coerenza dimensionale e NaN check [web:39][web:22]
    % assert(all(cellfun(@(x) size(x,1)==sequenceLength, XTrain))); % timesteps [web:39][web:22]
    % assert(size(YTrain,1)==numSequences);                         % N allineato [web:39][web:22]
    % assert(~any(isnan(YTrain(:))));                               % evitare NaN in Y [web:39][web:22]
end
