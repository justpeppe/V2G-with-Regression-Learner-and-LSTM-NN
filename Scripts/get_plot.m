function figure_out = get_plot(Ypred_norm, Ynorm, params, target, time_vector)
    % net: rete LSTM allenata
    % Xnorm: dati di input normalizzati (ad esempio X di training)
    % Ynorm: output reale normalizzato (ground truth)
    % params: struttura con campi media e dev_std per de-normalizzazione
    % target: nome del campo in params per media e dev_std
    % time_vector: vettore temporale per asse x nel plot
    
    % Predizione sulla base degli input normalizzati
    %Ypred_norm = predict(net, Xnorm);

    % De-normalizzazione
    Y_real = Ynorm .* params.(target).dev_std + params.(target).media;
    Y_pred = Ypred_norm .* params.(target).dev_std + params.(target).media;
    
    % Allinea il vettore temporale all'input (shiftato di un passo indietro)
    time_vector_input = time_vector(1:end-1);

    % Taglia le serie di una posizione per allinearle
    Y_real_aligned = Y_real(1:end-1);
    Y_pred_aligned = Y_pred(2:end);

    figure_out = figure;
    plot(time_vector_input, Y_real_aligned, 'k-', 'LineWidth', 1);
    hold on;
    plot(time_vector_input, Y_pred_aligned, 'r--', 'LineWidth', 1);
    legend('Dati Reali', 'Dati Predetti');
    title('Confronto Predizione vs Reali de-normalizzati (Allineato)');
    xlabel('Tempo');
    ylabel('Energia (kWh)');
    grid on;
    hold off;
end
