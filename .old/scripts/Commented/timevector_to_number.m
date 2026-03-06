function T_out = timevector_to_number(T_in)


% Copia la tabella di input per evitare di modificare l'originale
T_out = T_in;

% --- 1. Rimuovi la colonna 'datetime' se esiste ---
if any(strcmp(T_out.Properties.VariableNames, 'datetime'))
    T_out = removevars(T_out, 'datetime');
    disp('Colonna "datetime" rimossa.');
else
    disp('Colonna "datetime" non trovata.');
end

giorno_numerico = day(T_out.time_vector, 'iso-dayofweek');

T_out = addvars(T_out, giorno_numerico, 'Before', 1, 'NewVariableNames', 'giorno_settimana');
disp('Inserita colonna "giorno_settimana" con valore numerico da 1 a 7 (lunedi a domenica)');