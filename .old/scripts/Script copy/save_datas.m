function save_datas(tabellaDaSalvare, root, periodo, nomeBaseFile)
% save_datas Salva una tabella in formati .mat e .xlsx in una cartella basata sulla data.
%
% INPUT:
%   - tabellaDaSalvare: La tabella MATLAB che si desidera salvare.
%   - root:             Il percorso della cartella principale del progetto (stringa).
%   - periodo:          Una stringa, 'giorno' o 'notte', per nominare il file.
%   - nomeBaseFile:     Il nome base per il file di output (es. 'RegressionLearnerResults').

    % 1. Validazione dell'input 'periodo'
    if ~strcmpi(periodo, 'giorno') && ~strcmpi(periodo, 'notte')
        error("Il parametro 'periodo' deve essere 'giorno' o 'notte'.");
    end

    % 2. Creazione del percorso di salvataggio basato sulla data corrente
    current_date_str = datestr(now, 'yyyy_mm_dd');
    save_folder = fullfile(root, 'Sessioni', current_date_str);

    % Assicura che la cartella di destinazione esista
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
        disp(['Cartella creata: ', save_folder]);
    end

    % 3. Creazione dinamica dei nomi dei file
    file_name_mat = sprintf('%s_%s.mat', nomeBaseFile, periodo);
    file_name_xlsx = sprintf('%s_%s.xlsx', nomeBaseFile, periodo);

    % Crea i percorsi completi per i file
    full_path_mat = fullfile(save_folder, file_name_mat);
    full_path_xlsx = fullfile(save_folder, file_name_xlsx);

    % 4. Salvataggio dei file
    % Salva la variabile passata come input nel file .mat
    % NOTA: Il nome della variabile nel file .mat sarà 'tabellaDaSalvare'.
    save(full_path_mat, 'tabellaDaSalvare');
    
    % Salva la stessa tabella in formato Excel
    writetable(tabellaDaSalvare, full_path_xlsx);

    disp(['File salvati con successo in: ' save_folder]);
end
