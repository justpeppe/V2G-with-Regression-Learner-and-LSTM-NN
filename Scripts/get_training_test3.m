function [training, test] = get_training_test3(datas, test_days)
    
    % Verifica che esista la variabile time_vector e che sia datetime
    assert(ismember('time_vector', datas.Properties.VariableNames), ...
        'La tabella deve contenere la variabile time_vector.'); % controllo presenza
    if ~isdatetime(datas.time_vector)
        error('tabella.time_vector deve essere di tipo datetime.'); % tipo corretto
    end
    
    % Allinea tutti i datetime all'inizio del giorno per confronti coerenti
    temp = dateshift(datas.time_vector, 'start', 'day'); % normalizza righe
    test_days = dateshift(datetime(test_days), 'start', 'day'); % normalizza test

    idx_test = ismember(temp, test_days); % indice test

    % Per Regression Learner: nessun validation esplicito, solo training e test
    training = datas(~idx_test, :); % tutto tranne i giorni di test
    test = datas(idx_test, :); % solo giorni di test
    disp('Create tabelle Training e Test per Regression Learner');

end