function figure_regression_learner = get_plot_RL(colonna_dati_predetti, colonna_dati_reali, parametri_normalizzazione, target, time_vector)
    % get_plot_RL: De-normalizza vettori, grafica predizione vs realtà nel tempo
    %
    % colonna_dati_predetti: valori stimati (normalizzati)
    % colonna_dati_reali: valori reali (normalizzati)
    % parametri_normalizzazione: struct con .media e .dev_std per ciascun target
    % target: nome della variabile target (stringa)
    % time_vector: vettore datetime, stesso numero righe degli altri vettori

    % De-normalizza i dati predetti (operazione inversa dello z-score)
    dati_predetti_denorm = (colonna_dati_predetti * parametri_normalizzazione.(target).dev_std) + parametri_normalizzazione.(target).media;
    
    % De-normalizza i dati reali
    dati_reali_denorm = (colonna_dati_reali * parametri_normalizzazione.(target).dev_std) + parametri_normalizzazione.(target).media;

    Y_real_aligned = dati_reali_denorm(1:end-1);
    Y_pred_aligned = dati_predetti_denorm(2:end);
    time_vector_input = time_vector(1:end-1);
    
    % Crea la figura per la visualizzazione
    figure_regression_learner = figure;

    % Plot linea dati reali
    plot(time_vector_input, Y_real_aligned, 'k-', 'LineWidth', 1);       % Linea nera
    hold on;                                                            % Sovrappone altro plot

    % Plot linea dati predetti
    plot(time_vector_input, Y_pred_aligned, 'r--', 'LineWidth', 1);      % Linea blu

    % Aggiunge leggenda
    legend('Dati Reali', 'Dati Predetti');

    % Titolo del grafico
    title('Confronto Predizione vs Reali de-normalizzati (Allineato)');

    % Etichette assi
    xlabel('Tempo');
    ylabel('Energia (kWh)');

    % Migliora la griglia
    grid on;

end
