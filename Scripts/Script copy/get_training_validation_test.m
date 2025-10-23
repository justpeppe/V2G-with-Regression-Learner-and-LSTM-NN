function [training, validation, test] = get_training_validation_test(tabella, validation_days, test_days, model)
    
    % Converte il modello in string per un confronto robusto
    model = string(model); % garantisce uno string scalar

    % Verifica che esista la variabile time_vector e che sia datetime
    assert(ismember('time_vector', tabella.Properties.VariableNames), ...
        'La tabella deve contenere la variabile time_vector.'); % controllo presenza
    if ~isdatetime(tabella.time_vector)
        error('tabella.time_vector deve essere di tipo datetime.'); % tipo corretto
    end

    % Allinea tutti i datetime all'inizio del giorno per confronti coerenti
    temp = dateshift(tabella.time_vector, 'start', 'day'); % normalizza righe
    validation_days = dateshift(datetime(validation_days), 'start', 'day'); % normalizza validation
    test_days = dateshift(datetime(test_days), 'start', 'day'); % normalizza test

    % Costruisce gli indici per validation e test
    idx_validation = ismember(temp, validation_days); % indice validation
    idx_test = ismember(temp, test_days); % indice test

    % Risolve sovrapposizioni assegnando priorità al test
    idx_validation = idx_validation & ~idx_test; % rimuove giorni di test da validation

    % Inizializza sempre validation come tabella vuota (stesse colonne)
    validation = tabella([],:); % evita "output non assegnato"

    % Ramificazione in base al modello
    if model == "Regression Learner"
        % Per Regression Learner: nessun validation esplicito, solo training e test
        training = tabella(~idx_test, :); % tutto tranne i giorni di test
        test = tabella(idx_test, :); % solo giorni di test
        disp('Create tabelle Training e Test per Regression Learner');
    else
        % Per altri modelli: usa training, validation e test separati
        training = tabella(~idx_validation & ~idx_test, :); % esclude validation e test
        validation = tabella(idx_validation, :); % giorni di validation
        test = tabella(idx_test, :); % giorni di test
        disp('Create tabelle Training, Validation e Test per LSTM NN');
    end
end
