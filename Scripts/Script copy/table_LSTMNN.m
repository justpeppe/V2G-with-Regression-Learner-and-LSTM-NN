function [XTrain, TTrain] = table_LSTMNN(table, num_predittori, columns_table, target)
    % Estrae i dati numerici dai colonne specificate
    datas = table2array(table(:, columns_table));
    
    sequenceLength = num_predittori;
    nRows = size(datas, 1);
    
    % Calcolo corretto del numero di sequenze per non eccedere
    numSequences = nRows - sequenceLength; 
    
    XTrain = cell(numSequences, 1);
    for i = 1:numSequences
        % Ogni sequenza è trasposta per avere [features x timesteps]
        XTrain{i} = datas(i:i+sequenceLength-1, :); 
    end
    
    targetData = table.(target);
    TTrain = cell(numSequences, 1);
    for i = 1:numSequences
        % Target è il valore subito dopo la sequenza (step predittivo)
        TTrain{i} = targetData(i + sequenceLength, :);
    end
    
    % Converte TTrain in vettore numerico
    TTrain = cell2mat(TTrain);
    
    % Mantieni XTrain come cell array per il training LSTM
end
