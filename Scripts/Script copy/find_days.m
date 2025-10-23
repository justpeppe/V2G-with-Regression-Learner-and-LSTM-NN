function find_days(table, day_numeric)
    table_in = table; % copia tabella
    
    idx = find(table_in.giorno_settimana == day_numeric);
    table_out = table_in(idx, :);

    temp = dateshift(table_out.time_vector, 'start', 'day'); % normalizza righe
    unique(temp)
end

