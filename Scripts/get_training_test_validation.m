function [training, validation, test] = get_training_test_validation(t_input, validation_days, test_days)
    
    % Verifica che esista la variabile time_vector e che sia datetime
    assert(ismember('time_vector', t_input.Properties.VariableNames), ...
        'La tabella deve contenere la variabile time_vector.'); % controllo presenza
    if ~isdatetime(t_input.time_vector)
        error('tabella.time_vector deve essere di tipo datetime.'); % tipo corretto
    end
    
    % Allinea tutti i datetime all'inizio del giorno per confronti coerenti
    temp = dateshift(t_input.time_vector, 'start', 'day'); % normalizza righe
    test_days = dateshift(datetime(test_days), 'start', 'day'); % normalizza test
    validation_days = dateshift(datetime(validation_days), 'start', 'day'); % normalizza test

    idx_test = ismember(temp, test_days); % indice test
    idx_validation = ismember(temp, validation_days); % indice validation

    % Per Regression Learner: nessun validation esplicito, solo training e test
    training = t_input(~idx_test & ~idx_validation, :); % tutto tranne i giorni di test e validation
    validation = t_input(idx_validation, :); % solo giorni di validation
    test = t_input(idx_test, :); % solo giorni di test

    disp('Create tabelle Training, Validation e Test per LSTM NN');

end