function analize_datas(tabellaRisultati, colonneDaAnalizzare)
% analize_datas Ordina una tabella di risultati per diverse metriche,
% ignorando i modelli con valori NaN, e stampa un report del miglior modello.
%
% INPUT:
%   - tabellaRisultati:    La tabella di input (es. RegressionLearnerResults_notte).
%   - colonneDaAnalizzare: Un array numerico con gli indici delle colonne (es. 5:12)
%                          o una cella di stringhe con i nomi.

    % Se l'input è numerico, converti gli indici in nomi di colonne.
    if isnumeric(colonneDaAnalizzare)
        nomiColonne = tabellaRisultati.Properties.VariableNames(colonneDaAnalizzare);
    else
        nomiColonne = colonneDaAnalizzare;
    end
    
    fprintf('--- REPORT DEI MIGLIORI MODELLI (con esclusione di NaN) ---\n\n');

    % Ciclo su ogni metrica (colonna) da analizzare
    for i = 1:numel(nomiColonne)
        nomeColonnaCorrente = nomiColonne{i};

        % 1. FILTRAGGIO DEI NaN: Rimuovi le righe con NaN nella colonna corrente.
        dati_validi = tabellaRisultati(~isnan(tabellaRisultati.(nomeColonnaCorrente)), :);

        % 2. CONTROLLO DI SICUREZZA: Verifica se rimangono dati validi.
        if isempty(dati_validi)
            fprintf('Miglior modello secondo la metrica: "%s"\n', nomeColonnaCorrente);
            fprintf('-> Nessun modello valido trovato (tutti i valori sono NaN).\n');
            fprintf('---------------------------------------------------\n\n');
            continue; % Passa alla metrica successiva
        end
        
        % 3. DETERMINA L'ORDINE: Decrescente per R-Squared, crescente per gli altri.
        if contains(nomeColonnaCorrente, 'RSquared', 'IgnoreCase', true)
            ordine = 'descend';
        else
            ordine = 'ascend';
        end

        % 4. ORDINA E SELEZIONA: Ordina i dati validi e prendi il migliore.
        tabellaOrdinata = sortrows(dati_validi, nomeColonnaCorrente, ordine);
        migliorModello = tabellaOrdinata(1, :);

        % --- Stampa del report per questa metrica ---
        fprintf('Miglior modello secondo la metrica: "%s" (Ordine: %s)\n', nomeColonnaCorrente, ordine);
        
        modelNumber = migliorModello.("Model Number");
        modelType = migliorModello.("Model Type");
        
        fprintf('-> Il modello identificato è il n. %s, di tipo "%s".\n', string(modelNumber), string(modelType));
        disp('Dettagli completi:');
        disp(migliorModello);
        fprintf('---------------------------------------------------\n\n');
    end
end
