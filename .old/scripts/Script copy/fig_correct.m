function fig_correct (fig_path, time_vector_completo, parametri_norm)

openfig(fig_path); % apriamo la figura
dati_linee = get(gca, 'Children'); % salviamo i dati delle curve
dati_y = get(dati_linee, 'YData'); % salviamo i dati dell'asse y
dati_reali_norm = dati_y{1}; % i dati reali sono nella prima curva
dati_predetti_norm = dati_y{2}; % i dati predetti sono nella seconda curva
media = parametri_norm.AAC_energy.media; %preleviamo la media utilizzata in precedenza per normalizzare
dev_std = parametri_norm.AAC_energy.dev_std; % uguale per la varianza
dati_reali = (dati_reali_norm * dev_std) + media; % formula inversa
dati_predetti = (dati_predetti_norm * dev_std) + media;


figure; 
plot(time_vector_completo, dati_reali, 'DisplayName', 'Dati Reali');
hold on; 
plot(time_vector_completo, dati_predetti, 'DisplayName', 'Dati Predetti');
hold off;
xlabel('Tempo');     
ylabel('Energia (kWh)');    
title('Confronto tra Dati Reali e Dati Predetti'); 
legend('show');
grid on;