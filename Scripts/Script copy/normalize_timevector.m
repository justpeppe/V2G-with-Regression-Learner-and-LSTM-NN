function T_out = normalize_timevector(T_in)
% normalize_timevector Elabora una tabella per normalizzare un vettore temporale.
%
%   T_out = normalize_timevector(T_in) accetta una tabella T_in e restituisce
%   una tabella T_out in cui:
%   1. La colonna 'datetime' è stata rimossa.
%   2. La colonna 'time_vector' (di tipo datetime) è stata convertita in un
%      vettore numerico (secondi dall'istante iniziale).
%   3. La colonna 'time_vector' è stata normalizzata nell'intervallo [0, 1].

% Copia la tabella di input per evitare di modificare l'originale
T_out = T_in;

% --- 1. Rimuovi la colonna 'datetime' se esiste ---
if any(strcmp(T_out.Properties.VariableNames, 'datetime'))
    T_out = removevars(T_out, 'datetime');
    disp('Colonna "datetime" rimossa.');
else
    disp('Colonna "datetime" non trovata.');
end

% --- 2. Converti la colonna 'time_vector' da datetime a numerico ---
if any(strcmp(T_out.Properties.VariableNames, 'time_vector')) && isdatetime(T_out.time_vector)
    % Calcola i secondi trascorsi dal primo elemento
    time_numeric = seconds(T_out.time_vector - T_out.time_vector(1));
    
    % Sostituisci la colonna originale con i valori numerici
    T_out.time_vector = time_numeric;
    disp('Colonna "time_vector" convertita in formato numerico (secondi).');
    
    % --- 3. Normalizza la colonna 'time_vector' ---
    min_val = min(T_out.time_vector);
    max_val = max(T_out.time_vector);
    
    % Applica la normalizzazione min-max per scalare i valori tra 0 e 1
    if max_val > min_val
        T_out.time_vector = (T_out.time_vector - min_val) / (max_val - min_val);
        disp('Colonna "time_vector" normalizzata.');
    else
        % Se tutti i valori sono uguali, la normalizzazione non è necessaria
        % o potrebbe risultare in NaN. Impostiamo a 0 o a un valore costante.
        T_out.time_vector(:) = 0;
        disp('Tutti i valori in "time_vector" sono identici. Normalizzati a 0.');
    end
else
    error('La tabella non contiene una colonna "time_vector" di tipo datetime.');
end

end
