function tabella_output = create_lags(dati_input, dimensioneFinestra, nomiPredittori, nomeTarget)
% creaFeatureFinestraScorrevole Trasforma una serie storica in un dataset supervisionato
% utilizzando una finestra scorrevole.
%
% INPUT:
%   - dati_input:        Tabella di input contenente i dati temporali.
%   - dimensioneFinestra:  Numero di passi temporali passati (lag) da usare come feature.
%   - nomiPredittori:    Cell array con i nomi delle colonne da usare come predittori esogeni.
%   - nomeTarget:        Stringa con il nome della colonna da predire (es. 'AAC_energy').
%
% OUTPUT:
%   - tabella_output:    Tabella finale pronta per il training di un modello,
%                        con feature lag e una colonna target.

    %% Estrazione dati dalla tabella di input
    predittoriInput = dati_input{:, nomiPredittori};
    targetInput = dati_input.(nomeTarget);

    %% Calcolo dimensioni
    numeroCampioniTotali = height(dati_input);
    numeroPredittori = numel(nomiPredittori);
    numeroNuoveRighe = numeroCampioniTotali - dimensioneFinestra;

    % Il numero di colonne è dato da: (1 colonna target + N predittori) * dimensioneFinestra
    numeroFeatureLag = (1 + numeroPredittori) * dimensioneFinestra;

    %% Preallocazione delle matrici per efficienza
    matriceFeature = zeros(numeroNuoveRighe, numeroFeatureLag);
    vettoreTarget = zeros(numeroNuoveRighe, 1);

    % Il vettore tempo di output parte dal primo campione dopo la prima finestra
    time_out = dati_input.time_vector(dimensioneFinestra + 1 : end);

    %% Creazione delle feature con la finestra scorrevole
    for i = 1:numeroNuoveRighe
        % Indici per la finestra corrente nel dataset originale
        indiceFineFinestra = i + dimensioneFinestra - 1;

        % Estrai i dati della finestra per il target e i predittori
        finestra_target = targetInput(i : indiceFineFinestra);
        finestra_predittori = predittoriInput(i : indiceFineFinestra, :);

        % "Appiattisci" la finestra in una singola riga di feature
        % L'ordine è [target_t-1, pred1_t-1, ..., target_t-2, pred1_t-2, ...]
        riga_feature = [finestra_target'; finestra_predittori'];
        matriceFeature(i, :) = riga_feature(:)'; % Il trasposto finale allinea i dati per riga

        % Il target è il valore subito dopo la fine della finestra
        vettoreTarget(i) = targetInput(indiceFineFinestra + 1);
    end

    %% Creazione dinamica dei nomi delle colonne
    nomiColonne = cell(1, numeroFeatureLag);
    col_idx = 1;
    % L'ordine dei nomi deve corrispondere all'appiattimento di sopra
    for k = 1:dimensioneFinestra
        % Nome per il target (es. Energia_t-1)
        nomiColonne{col_idx} = sprintf('%s_t-%d', nomeTarget, k);
        col_idx = col_idx + 1;
        % Nomi per i predittori (es. windspeed_t-1)
        for p = 1:numeroPredittori
            nomiColonne{col_idx} = sprintf('%s_t-%d', nomiPredittori{p}, k);
            col_idx = col_idx + 1;
        end
    end

    %% Assemblaggio della tabella finale
    tabella_output = array2table([matriceFeature, vettoreTarget], ...
        'VariableNames', [nomiColonne, {strcat(nomeTarget, '_Target')}]);

    % Aggiunge il vettore temporale normalizzato come prima colonna
    tabella_output = addvars(tabella_output, time_out, 'Before', 1, 'NewVariableNames', 'time_vector');

    %disp('Tabella con feature a finestra scorrevole creata con successo.');
end
