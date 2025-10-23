function t_output = is_zero(t_input)
    
    % Copia la tabella di input in una variabile locale
    t = t_input;
    
    % Crea una variabile logica che identifica dove AAC_energy è zero
    idx_zero = t.AAC_energy == 0;
    
    % Inizializza la nuova colonna is_zero con tutti zeri
    t.is_zero = zeros(height(t), 1);
    
    % Assegna il valore 1 alle righe dove AAC_energy è zero
    t.is_zero(idx_zero) = 1;
    
    % Restituisce la tabella modificata
    t_output = t;

    disp('Aggiunta colonna is_zero con valore 1 (true) quando AAC_energy è zero');
end
