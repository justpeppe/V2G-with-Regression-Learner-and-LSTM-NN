function T_out = timevector_hour_to_number(T_in)

% Copia la tabella di input per evitare di modificare l'originale
T_out = T_in;

time_vector = T_out.time_vector;
numericTime = hour(time_vector) + minute(time_vector)/60 + second(time_vector)/3600;

T_out = addvars(T_out, numericTime, 'Before', 1, 'NewVariableNames', 'orario_numerico');
disp('Inserita colonna "orario_numerico" con valore numerico da 0 a 24 (.5 = mezz"ora)');