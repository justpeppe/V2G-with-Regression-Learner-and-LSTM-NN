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

    % Plot confronto dati reali vs predetti con time_vector sull'asse x
    figure_out = figure;
    plot(time_vector, Y_real, 'k-', 'LineWidth', 1); % nero linea continua
    hold on;
    plot(time_vector, Y_pred, 'r--', 'LineWidth', 1); % rosso linea tratteggiata
    legend('Dati Reali', 'Dati Predetti');
    title('Confronto Predizione vs Reali de-normalizzati');
    xlabel('Tempo');
    ylabel('Energia (kWh)');
    grid on;
    hold off;
end
