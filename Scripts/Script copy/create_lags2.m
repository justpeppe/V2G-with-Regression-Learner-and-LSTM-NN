function tabella_output = create_lags2(dati_input, dimensioneFinestra, nomiPredittori, nomeTarget, ordine)
    % Crea un dataset supervisionato one-step-ahead: lags t-1..t-H e predittori correnti a t.
    % ordine: "descend" => colonne in ordine t-1, t-2, ..., t-H (default)
    %         "ascend"  => colonne in ordine t-H, ..., t-2, t-1

    if nargin < 5 || isempty(ordine)
        ordine = "ascend";                                    % default ordinamento lag
    end
    assert(ismember(ordine, ["descend","ascend"]), ...
        'ordine deve essere "descend" o "ascend".');           % valida parametro ordine

    X = dati_input{:, nomiPredittori};                         % matrice predittori (N x P)
    y = dati_input.(nomeTarget);                               % vettore target (N x 1)
    N = height(dati_input);                                    % numero righe
    P = numel(nomiPredittori);                                 % numero predittori
    H = dimensioneFinestra;                                    % ampiezza finestra (numero di lag)

    M = N - H - 1;                                             % numero righe output (serve target a t+1)
    if M <= 0
        error('dimensioneFinestra troppo grande rispetto alla lunghezza dei dati.');
    end

    F_lag = zeros(M, (1+P)*H);                                 % feature lag (target + predittori) solo passate
    F_cur = zeros(M, P);                                       % predittori correnti a t
    y_out = zeros(M, 1);                                       % target futuro a t+1
    t_out = dati_input.time_vector((H+2):end);                 % time allineato a t+1

    for r = 1:M                                                % riga di output
        c = r + H;                                             % indice "corrente" t
        lag_idx = (c-H):(c-1);                                 % indici t-H .. t-1 (esclude t)
        if ordine == "descend"
            lag_idx = fliplr(lag_idx);                         % t-1, t-2, ..., t-H
        end

        col = 1;                                               % indice di colonna nella matrice lag
        for k = 1:H                                            % per ciascuna lag
            tk = lag_idx(k);                                   % indice del tempo per questa lag
            F_lag(r, col) = y(tk);                             % target lag t-k
            col = col + 1;
            for p = 1:P                                        % predittori lag t-k
                F_lag(r, col) = X(tk, p);
                col = col + 1;
            end
        end

        F_cur(r, :) = X(c, :);                                 % predittori correnti a t
        y_out(r)    = y(c + 1);                                % target futuro a t+1
    end

    % Nomi colonne lag: per ciascun k, prima target_t-k poi ogni predittore_t-k
    names_lag = cell(1, (1+P)*H);                              % prealloc nomi lag
    idx = 1;                                                   % indice nomi
    for k = 1:H                                                % lag 1..H
        lag_k = k;                                             % etichetta t-k
        names_lag{idx} = sprintf('%s_t-%d', nomeTarget, lag_k);
        idx = idx + 1;
        for p = 1:P
            names_lag{idx} = sprintf('%s_t-%d', nomiPredittori{p}, lag_k);
            idx = idx + 1;
        end
    end

    % Nomi colonne per predittori correnti
    names_cur = strcat("current_", string(nomiPredittori));    % current_ per predittori

    % Costruzione tabella finale: lag + current_pred + target futuro
    tabella_output = array2table([F_lag, F_cur, y_out], ...
        'VariableNames', [names_lag, cellstr(names_cur), {strcat(nomeTarget, '_Target')}]);

    % Aggiunge time_vector allineato al target (t+1) come prima colonna
    tabella_output = addvars(tabella_output, t_out, 'Before', 1, 'NewVariableNames', 'time_vector');

    % Non includiamo il target corrente: non è mai stato aggiunto, quindi niente da rimuovere
    % Nota: in Regression Learner seleziona come risposta <nomeTarget>_Target
    % ed escludi time_vector dai predittori se non vuoi usarlo come feature.
end
