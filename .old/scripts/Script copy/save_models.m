function save_models(root, models, varName)
    rootFolder = root; % inserisci qui la root (es. 'C:\MieiProgetti')
    currentDate = datestr(now, 'yyyy_mm_dd'); % formato YYYY-MM-DD
    %currentTime = datestr(now, 'HH_MM');      % formato HH-MM
    
    % Costruisci i percorsi di cartella
    dateFolder = fullfile(rootFolder, 'Sessioni', currentDate);
    timeFileName = ['Models_' varName '.mat'];
    fullPath = fullfile(dateFolder, timeFileName);
    
    % Crea cartella dataFolder se non esiste
    if ~exist(dateFolder, 'dir')
        mkdir(dateFolder);
    end
    
    % Salva la variabile models nel file specificato
    save(fullPath, 'models');
    
    disp(['Modello salvato in: ' fullPath]);

end