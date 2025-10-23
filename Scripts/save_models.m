function save_models(rootFolder, models, varName, current_net)
% save_models — Crea/usa cartella rootFolder/sessioni/YYYY_mm_dd e file Models_YYYY_mm_dd.mat; accoda il modello
%
% INPUT:
%   rootFolder  : stringa, cartella radice (es. 'C:\MieiProgetti')
%   models      : struct, contiene il nuovo modello in models.(varName)
%   varName     : stringa, nome chiave del modello (es. 'net_2025_10_12_13_51')
%   current_net : oggetto/rete addestrata da salvare come 'current_net' nel struct

    currentDate = datestr(now, 'yyyy_mm_dd');                    % Data corrente in formato YYYY_mm_dd
    dateFolder = fullfile(rootFolder, 'sessioni', currentDate);  % Percorso cartella: root/sessioni/YYYY_mm_dd
    if ~exist(dateFolder, 'dir')                                 % Se la cartella non esiste
        mkdir(dateFolder);                                       %   creala
    end                                                          % Fine creazione cartella

    matFile = fullfile(dateFolder, ['Models_' currentDate '.mat']); % Nome file MAT giornaliero
                                                    % Fine normalizzazione campo

    
    if ~isempty(current_net)           % Se esiste un parametro valido
        models.(varName).net = current_net;                          % imposta/aggiorna il campo
    end                                                              % Fine allineamento parametro

    % Carica eventuali modelli già presenti nel file del giorno
    if exist(matFile, 'file')                                        % Se il file esiste
        loaded = load(matFile, 'models');                            %   carica la variabile 'models'
        if isfield(loaded, 'models') && isstruct(loaded.models)      %   se la struttura è presente
            existing = loaded.models;                                %     mantieni i modelli esistenti
        else                                                         %   altrimenti
            existing = struct();                                     %     parti da uno struct vuoto
        end                                                          % Fine check struttura esistente
    else                                                             % Se il file non esiste
        existing = struct();                                         %   crea uno struct vuoto da popolare
    end                                                              % Fine gestione caricamento

    % Accoda/sovrascrive il modello corrente sotto la chiave varName
    existing.(varName) = models.(varName);                           % Inserisci/aggiorna il modello della sessione
    models = existing;                                               % Aggiorna la variabile complessiva da salvare

    % Salvataggio robusto (v7.3 per grandi reti/dataset)
    save(matFile, 'models', '-v7.3');                                % Salva l'intera struttura modelli nel file
    fprintf('Modello "%s" salvato in: %s\n', varName, matFile);      % Messaggio di conferma percorso
end
