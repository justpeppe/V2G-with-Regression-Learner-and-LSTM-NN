function [datas_norm, parametri_norm] = normalize_zscore_2(datas, colonne_da_normalizzare)
% normalize_zscore_2: normalizza colonne specifiche (numeric, datetime, duration)
% usando z-score. Parametri calcolati sull'intero dataset 'datas'.
%
% Input:
%  - datas: La tabella MATLAB che contiene tutti i dati.
%  - colonne_da_normalizzare: Cell array di nomi di colonna da normalizzare.
%
% Output:
%  - datas_norm: Tabella completa con le colonne specificate normalizzate.
%  - parametri_norm: struct con media e dev_std calcolate sull'intero dataset.


    % 1) Validazione e Preparazione dei Dati
    assert(istable(datas), 'L''input ''datas'' deve essere una tabella MATLAB.');
   
    % Assicura che colonne_da_normalizzare sia cell array
    if isstring(colonne_da_normalizzare)
        colonne_da_normalizzare = cellstr(colonne_da_normalizzare);
    elseif ischar(colonne_da_normalizzare)
        colonne_da_normalizzare = {colonne_da_normalizzare};
    end
    assert(iscellstr(colonne_da_normalizzare), ...
        'colonne_da_normalizzare deve essere cell array di nomi di variabile.'); 
    
    % Verifica esistenza colonne
    assert(all(ismember(colonne_da_normalizzare, datas.Properties.VariableNames)), ...
        'Alcune colonne non esistono nella tabella datas.');
    
    % 2) Inizializzazione
    parametri_norm = struct();
    % Inizializza datas_norm come una copia di datas per mantenere tutte le altre colonne
    datas_norm = datas; 
    
    % 3) Loop colonne da normalizzare
    for i = 1:numel(colonne_da_normalizzare)
        col_nome = colonne_da_normalizzare{i};                             
        v_full = datas.(col_nome); % <- DATI COMPLETI: Sostituisce v_train, v_test
        
        % Tipi ammessi: numeric, datetime, duration
        if ~(isnumeric(v_full) || isdatetime(v_full) || isduration(v_full))
            error('La colonna %s non è numeric/datetime/duration: impossibile normalizzare.', col_nome);
        end
        
        % 4) Calcolo media (mu) e dev std (sg) sull'INTERO dataset
        if isdatetime(v_full)
            mu = mean(v_full, 'omitnat');                                      
            sg = std(v_full, 0, 'omitnat');                                    
            sg_is_zero = (seconds(sg) == 0) | isnan(seconds(sg));               
            if sg_is_zero, sg = seconds(1); end                                 
        elseif isduration(v_full)
            mu = mean(v_full, 'omitnan');                                      
            sg = std(v_full, 0, 'omitnan');                                    
            sg_is_zero = (seconds(sg) == 0) | isnan(seconds(sg));               
            if sg_is_zero, sg = seconds(1); end                                 
        else % Numerico
            mu = mean(v_full, 'omitnan');                                      
            sg = std(v_full, 0, 'omitnan');                                    
            if isnan(sg) || sg == 0, sg = 1; end                                
        end
        
        % 5) Applicazione Z-score all'intero set di dati
        z_full = (v_full - mu) ./ sg; % Normalizza l'intera colonna
        
        % 6) Scrive output normalizzato e parametri
        datas_norm.(col_nome) = z_full;                                 
        
        safe_field = matlab.lang.makeValidName(col_nome);                   
        parametri_norm.(safe_field).media  = mu;                            
        parametri_norm.(safe_field).dev_std = sg;                           
    end
    
    disp('Normalizzazione Z-score completata sull''intero dataset.');
end