function [training, test] = get_training_and_test(tabella_modello, giorno_test1, giorno_test2)

% Inizializzazione tabelle di output (mantenendo la struttura)
training = tabella_modello(false, :);
test = tabella_modello(false, :);

% La funzione ismember controlla quali righe della colonna 'giorno_numerico' 
% hanno un valore uguale a giorno_test1 o giorno_test2.
idx_test = ismember(tabella_modello.giorno_settimana, [giorno_test1, giorno_test2]);

test = tabella_modello(idx_test, :);
training = tabella_modello(~idx_test, :);