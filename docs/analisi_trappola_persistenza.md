# Analisi del Problema della "Trappola della Persistenza" (Persistence Trap)

## Stato Attuale
Attualmente, l'architettura della rete neurale LSTM nel progetto prevede la predizione del consumo energetico (variabile `AAC_energy`) per l'istante immediatamente successivo all'ultimo campionamento disponibile nella finestra temporale di input. 

La generazione delle sequenze nel file `createLstmSequences.m` è così definita:
```matlab
tEnd = i + numPredictors - 1; % Fine della finestra dei predittori (es. 48 lag)
tTarget = tEnd + 1;           % L'istante da prevedere (Target, +30 minuti)
```

Date le caratteristiche dei dati, che presentano una frequenza di campionamento di 30 minuti, la differenza di assorbimento della rete elettrica tra un istante $t$ e l'istante $t+1$ è intrinsecamente trascurabile nella stragrande maggioranza dei casi. 

### Il Problema (Overfitting Architetturale)
Durante la fase di training, l'algoritmo di ottimizzazione (es. Adam) cerca di minimizzare l'errore quadratico medio (MSE). La rete apprende rapidamente che il modo matematicamente più "semplice" ed "economico" per azzerare l'errore a breve termine non è quello di mappare la complessa relazione non lineare tra meteo, giorno della settimana e stagionalità, bensì applicare una semplice funzione identità:
$$ y(t) \approx x(t-1) $$

Questo fenomeno si definisce come **Trappola della Persistenza** (o *Persistence Baseline Trap*).
Osservando i grafici comparativi (Predizione vs Reale), si nota visivamente come la curva della previsione neurale non sia altro che la curva reale traslata in avanti di uno step (uno shift di 30 minuti). 
Pertanto, nonostante metriche apparentemente eccellenti in fase di test (alti valori di $R^2$ e bassi RMSE), la rete non possiede alcun reale "potere predittivo" utile, in quanto sta banalmente copiando l'istante precedente.

## Impatto sul dominio applicativo (Vehicle-to-Grid)
Nel dominio del *Vehicle-to-Grid* (V2G), per poter bilanciare la rete elettrica ("frequency regulation", "peak shaving") o per schedulare la carica/scarica ottimale di un parco veicoli, è fondamentale conoscere e anticipare il profilo di carico atteso con largo anticipo (solitamente dalle 12 alle 24 ore prima).

Avere un modello che prevede, con uno scarto di soli 30 minuti, che il consumo sarà identico a quello attuale è un risultato insufficiente sotto il profilo ingegneristico, poiché non garantisce il tempo fisico necessario a comandare la flotta di veicoli elettrici per l'immissione o il prelievo di energia in rete.

---

## Soluzioni Proposte

Per ovviare al problema, occorre alterare la logica del processo (o "gioco") che la rete deve apprendere, spostando l'obiettivo da un banale nowcasting a una più sofisticata logica predittiva. Si propongono di seguito le principali strategie.

### Soluzione 1: Previsione dell'Orizzonte "Day-Ahead" (Previsione a 24 ore)
*L'opzione più indicata per lo use-case V2G.*
Anziché richiedere alla rete di prevedere il consumo tra mezz'ora, si sposta in avanti il target ("horizon") della previsione. Se i campioni storici forniti coprono fino alle ore 10:00 di "oggi", alla rete viene chiesto di prevedere il campione delle ore 10:00 di "domani" (ossia 48 campioni nel futuro).

**Vantaggi:** 
- La rete è costretta strutturalmente ad abbandonare la logica della persistenza, dato che il consumo di "domani" alle 10 non è quasi mai deducibile come pura e semplice copia di "oggi" alle 10. Questa distanza richiede necessariamente l'interpretazione del meteo futuro e del calendario (es. se domani è un giorno festivo e oggi è feriale).
- Risponde all'effettivo bisogno del mercato energetico elettrico (il *Day-Ahead Market* o MGP in Italia lavora esattamente su questa scala temporale).

**Implementazione (in `createLstmSequences.m`):**
```matlab
horizon = 48; % Orizzonte di 48 campioni (24 ore)
for i = 1:(numRows - numPredictors - horizon) % Prevenzione degli Out-of-bounds
    tEnd = i + numPredictors - 1; 
    tTarget = tEnd + horizon; % Shift target di +24h
    % ...
```

### Soluzione 2: Previsione della Variazione (Delta Differencing)
*Per mantenere un orizzonte a breve termine ma con valenza analitica.*
Si mantiene la previsione sul campione successivo (`tEnd + 1`), ma si modifica in pre-processing il target reale: la rete non dovrà più stimare il **valore assoluto** del consumo $y(t)$, ma il suo **delta** $\Delta y = y(t+1) - y(t)$. 

**Vantaggi:**
- Eliminando il trend di base assoluto, la rete per azzerare l'MSE deve necessariamente valutare la prima derivata della curva usando i parametri esogeni (meteo/orario), capendo se la curva è in flessione o in salita.

**Implementazione:**
- Costruzione di una nuova variabile pre-processata (es. `Delta_energy = [0; diff(datas.AAC_energy)]`) da impostare come `target` in input per il training e le pipeline di `createLstmSequences.m`.

### Soluzione 3: Architettura Sequence-to-Sequence (Seq2Seq)
*L'approccio matriciale multivariato.*
Invece di prevedere uno scalare isolato nello spazio temporale, si addestra la rete (tipicamente tramite una topologia *Encoder-Decoder LSTM* o impostando `lstmLayer` con `OutputMode="sequence"`) a ricevere in input un vettore di storia passata e restituire in output un vettore contiguo (es. le successive 24 ore globali ad intervalli di 30 minuti).

**Vantaggi:** 
- Rende disponibile istantaneamente all'algoritmo V2G l'intera curva attesa del giorno seguente, offrendo una granularità perfetta per l'ottimizzazione matematica della flotta.
**Svantaggi:**
- L'architettura neurale e la struttura di dimensionamento dell'input/output si complicano severamente. Richiede una totale re-ingegnerizzazione della logica di aggregazione dei dati, della loss function e dei layer fully connected in chiusura.

### Raccomandazione Finale
Per il corretto proseguimento del progetto e per valorizzare lo studio tecnico sul V2G in ambito tesi, si raccomanda l'adozione della **Soluzione 1 (Previsione Day-Ahead ad Orizzonte Distante)**. Essa bilancia eccellentemente il bisogno accademico/tecnico di debellare l'overfitting da persistenza con i reali bisogni di pianificazione energetica a lungo termine delle control room.
