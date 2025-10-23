function [test, validation] = test_and_validation(tabella_modello, giorni_validation)
    
    tbl = tabella_modello;
    if ~ismember('time_vector', tbl.Properties.VariableNames)
        error('La tabella deve contenere la colonna ''time_vector''.');             %
    end                                                                              %

    % time_vector deve essere datetime                                               %
    if ~isdatetime(tbl.time_vector)
        error('La colonna time_vector deve essere di tipo datetime.');               %
    end                                                                              %

    % Riduci ogni timestamp all'inizio del giorno (ignora l'orario)                 %
    d_tbl = dateshift(tbl.time_vector, 'start', 'day');                               

    % Normalizza/converti validationDays a datetime e porta al "giorno"             %
    if ~isdatetime(giorni_validation)
        try
            giorni_validation = datetime(giorni_validation);                                %
        catch ME
            error('validationDays non convertibile a datetime: %s', ME.message);      %
        end
    end
    d_val = dateshift(giorni_validation(:), 'start', 'day');

    % Confronto "solo data": appartenenza del giorno ai giorni di validation        %
    maskVal = ismember(d_tbl, d_val);                                                 % 
    maskTest = ~maskVal;                                                              %

    % Sottotabelle                                                                  %
    validation  = tbl(maskVal, :);                                                         %
    test = tbl(maskTest, :);                                                        %

end
