# Analisi Completa della Pipeline LSTM

> **Obiettivo**: Previsione della serie temporale `AAC_energy` (energy demand per EV charging zone) a partire da 48 campioni passati con 5 feature predittori.

---

## 1. Architettura della Pipeline

```
loadZoneData → splitTrainValTest → normalizeZScore → createLstmSequences → trainnet
```

---

## 2. Problemi Critici Identificati

### ❌ BUG #1 — Dimensione errata dell'Input Layer (CRITICO)

**File**: `LSTM.m`, riga 46–55

```matlab
% ATTUALE (ERRATO):
numFeatures = size(xTrain{1}, 2);  % restituisce il numero di COLONNE → 5 features
layers = [
    sequenceInputLayer(numFeatures, ...)  % aspetta: [numFeatures x numTimeSteps]
    ...
];
```

**Il problema**: `createLstmSequences` costruisce ogni sequenza come una matrice `[numLags × numFeatures]`, ovvero `[48 × 5]`. Questo significa che:
- `xTrain{i}` ha dimensione `[48, 5]` → righe = time steps, colonne = features
- `size(xTrain{1}, 2)` restituisce `5` (numero di colonne = features)
- MATLAB LSTM si aspetta input nel formato `[numFeatures × numTimeSteps]`, cioè `[5 × 48]`

**Il `sequenceInputLayer(5)` sembra corretto** — ma le sequenze sono orientate al contrario rispetto alla convenzione MATLAB.

La convenzione MATLAB per `lstmLayer` richiede che ogni sequenza sia una matrice `[C × T]` dove:
- `C` = numero di canali/feature
- `T` = numero di time step

Ma `xTrain{i}` è `[48 × 5]` (T×C), non `[5 × 48]` (C×T).

**Effetto**: La rete "funziona" perché MATLAB prova a interpretarla ugualmente, ma apprende pattern sbagliati: tratta le 5 feature come time steps e i 48 time steps come feature. Questo è devastante per le performance.

**Soluzione in `createLstmSequences.m`**: trasporre la matrice prima di salvarla:
```matlab
% RIGA 78 – da:
xTrainTemp{idxSequence} = xMat(i:tEnd, :);
% a:
xTrainTemp{idxSequence} = xMat(i:tEnd, :)';  % trasposizione! Shape: [features × lags]
```

---

### ❌ BUG #2 — Normalizzazione della validation set con parametri sbagliati (DATA LEAKAGE)

**File**: `LSTM.m`, riga 37–38

```matlab
% ATTUALE (POTENZIALE LEAKAGE):
[trainingNorm, testNorm, normParams]  = normalizeZScore(training, test, columnsToNormalize);
[~, validationNorm, ~]                = normalizeZScore(training, validation, columnsToNormalize);
```

Questa parte è **corretta** in principio (si normalizza sempre rispetto al training set), ma restituisce `normParams` calcolato *due volte* sullo stesso training set. Non c'è leakage qui, ma c'è **ridondanza**: i `normParams` prodotti dalla seconda chiamata sono identici alla prima poiché il training set è lo stesso.

**Problema sottile**: il `testNorm` e il `validationNorm` vengono calcolati correttamente, ma come viene usato `normParams` nel resto del codice? Se viene usato solo il `normParams` del primo `normalizeZScore`, la denormalizzazione funziona correttamente sia per train che test. ✅

---

### ❌ BUG #3 — `OutputMode="sequence"` nei layer LSTM intermedi è inefficiente/scorretto per questo task

**File**: `LSTM.m`, righe 56–60

```matlab
layers = [
    sequenceInputLayer(numFeatures, Normalization="none")
    lstmLayer(numHiddenUnits, OutputMode="sequence")      % → output [128 × 48]
    dropoutLayer(drop)
    lstmLayer(numHiddenUnits/2, OutputMode="sequence")    % → output [64 × 48]
    dropoutLayer(drop)
    lstmLayer(numHiddenUnits/4, OutputMode="last")        % → output [32 × 1]
    fullyConnectedLayer(numResponses)
];
```

Per un task di **regressione many-to-one** (48 input → 1 output), i layer intermedi con `OutputMode="sequence"` propagano l'intera sequenza al layer successivo (48 time steps), ma solo l'ultimo layer usa `"last"`. Questo è tecnicamente valido ma **introduce complessità non necessaria** e raddoppia/triplica il numero di parametri.

**Alternativa consigliata** (più efficiente e comprovata):
```matlab
layers = [
    sequenceInputLayer(numFeatures, Normalization="none")
    lstmLayer(128, OutputMode="last")   % estrae solo il contesto finale
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
];
```

---

### ⚠️ PROBLEMA #4 — Splitting del dataset: le sequenze attraversano i confini train/val/test

**File**: `LSTM.m`, riga 35 + `splitTrainValTest.m` + `createLstmSequences.m`

**Flusso attuale**:
1. `splitTrainValTest` separa i giorni di validation e test dal resto (training)
2. `createLstmSequences` viene chiamato **su ciascun split separatamente**

**Il problema nella validation set**: Il validation set contiene solo 2 giorni (21-22 Feb e 21-22 Luglio). Quando `createLstmSequences` cerca di costruire sequenze da quei 4 giorni (`numLags = 48` sample = 2 giorni interi a cadenza 30 min), la funzione verifica la **consecutività dei giorni** all'interno del set.

I giorni di validation (21-22 Feb, 21-22 Lug) **NON sono consecutivi tra loro**, quindi `createLstmSequences` può creare sequenze solo all'interno dei singoli blocchi di 2 giorni consecutivi. Con `numLags=48`, ogni blocco produce solo:
- `96 samples - 48 lags = 48 sequenze` per blocco ×2 blocchi = **~96 sequenze di validation**

Questo è un dataset di validation **estremamente piccolo** e poco rappresentativo.

**Problema più grave**: la sequenza di validation ha bisogno dei 48 campioni precedenti al primo campione di test per essere costruita correttamente — ma questi campioni vengono esclusi dal training set e non inclusi nel validation set. Le ultime 48 osservazioni del training prima del giorno di validation **mancano dalla sequenza di validation**. La funzione di validità (`isValid`) nelle sequenze attraverso giorni non consecutivi rifiuta queste sequenze.

**Soluzione**: includere un "contesto" di `numLags` campioni dal training set nel validation/test set, solo per permettere la costruzione delle sequenze.

---

### ⚠️ PROBLEMA #5 — `Shuffle="never"` per dati temporali è corretto ma limita la generalizzazione

**File**: `LSTM.m`, riga 67

```matlab
Shuffle="never"
```

Per serie temporali, `Shuffle="never"` è **corretto** in quanto preserva l'ordine temporale. Tuttavia, senza shuffling, il modello potrebbe non generalizzare bene a nuovi pattern. Si potrebbe usare `Shuffle="every-epoch"` se le sequenze sono sufficientemente diverse tra loro e indipendenti. Con la corretta trasposizione (BUG #1), lo shuffling diventa più sicuro.

---

### ⚠️ PROBLEMA #6 — `LearnRateDropPeriod=30` riferito alle epoche, non alle iterazioni

**File**: `LSTM.m`, riga 71

```matlab
LearnRateDropPeriod=30  % ogni 30 epoche, dimezza il LR
```

Con `MaxEpochs=150` il learning rate viene dimezzato 5 volte (dopo ep. 30, 60, 90, 120, 150). Partendo da `1e-3`, dopo 150 epoche il LR finale è:
```
1e-3 × (0.5)^5 = 3.125e-5
```
Questo potrebbe rendere il training troppo lento nella fase finale. Valutare di aumentare `LearnRateDropPeriod` a 50.

---

### ⚠️ PROBLEMA #7 — `computeMetrics` chiamata con dati non denormalizzati nel plot

**File**: `LSTM.m`, riga 136

```matlab
models.(netName).net_indicators.Test = computeMetrics(testPrediction, yTest, normParams, char(target));
```

La funzione `computeMetrics` riceve `yTest` (già in scala normalizzata) e lo denormalizza internamente. La funzione `plotResults` invece lavora su dati normalizzati e **dovrebbe** denormalizzare internamente anch'essa — ma questo dipende dall'implementazione di `plotResults.m`. Verificare la coerenza.

---

### ⚠️ PROBLEMA #8 — Manca l'`EarlyStopping`

**File**: `LSTM.m`, opzioni di training

Non è presente un criterio di **early stopping**. Con 150 epoche senza stop automatico, il modello potrebbe:
1. **Overfittare** se la validation loss risale dopo un minimo
2. **Sprecare tempo** di computazione dopo la convergenza

**Soluzione**:
```matlab
options = trainingOptions("adam", ...
    ...
    ValidationPatience=15, ...   % stop se val. loss non migliora per 15 check
    ...
);
```

---

## 3. Riepilogo Issue per Priorità

| # | Tipo | File | Severità | Impatto sulle Performance |
|---|------|------|----------|--------------------------|
| 1 | Bug | `createLstmSequences.m` | 🔴 CRITICO | Altissimo — la rete impara feature sbagliate |
| 2 | Data leakage | `LSTM.m` | 🟡 Basso | Nessuno (già corretto) |
| 3 | Architettura | `LSTM.m` | 🟠 Medio | Medio — modello sovracomplesso |
| 4 | Data split | `LSTM.m`/`splitTrainValTest.m` | 🟠 Medio | Medio — validation poco rappresentativa |
| 5 | Shuffle | `LSTM.m` | 🟡 Basso | Basso |
| 6 | LR Schedule | `LSTM.m` | 🟡 Basso | Basso |
| 7 | Metrics | `LSTM.m` | 🟡 Basso | Solo metriche, non training |
| 8 | EarlyStopping | `LSTM.m` | 🟠 Medio | Medio — rischio overfitting |

---

## 4. Correzioni Proposte

### 4.1 Fix BUG #1 — Trasposizione sequenze (PRIORITÀ MASSIMA)

In `createLstmSequences.m`, riga 78:
```matlab
% DA:
xTrainTemp{idxSequence} = xMat(i:tEnd, :);
% A:
xTrainTemp{idxSequence} = xMat(i:tEnd, :)';  % [numFeatures × numLags]
```

Dopo questo fix, `numFeatures = size(xTrain{1}, 1)` (non più colonna 2).
Aggiornare `LSTM.m`, riga 46:
```matlab
% DA:
numFeatures = size(xTrain{1}, 2);
% A:
numFeatures = size(xTrain{1}, 1);
```

### 4.2 Semplificare l'architettura (PRIORITÀ ALTA)

```matlab
layers = [
    sequenceInputLayer(numFeatures, Normalization="none")
    lstmLayer(128, OutputMode="last")
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
];
```

### 4.3 Aggiungere EarlyStopping (PRIORITÀ MEDIA)

```matlab
options = trainingOptions("adam", ...
    MaxEpochs=150, ...
    MiniBatchSize=mb, ...
    Shuffle="never", ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=50, ...
    GradientThreshold=1, ...
    GradientThresholdMethod="l2norm", ...
    L2Regularization=1e-4, ...
    ValidationData={xValidation, yValidation}, ...
    ValidationFrequency=valFreq, ...
    ValidationPatience=15, ...
    Metrics=["rsquared", "rmse", "mape", "mae", "mse"], ...
    Plots="training-progress", ...
    Verbose=false, ...
    ExecutionEnvironment="auto" ...
);
```

### 4.4 Strategia Split + Sequenze migliorata (PRIORITÀ MEDIA)

Per costruire sequenze di validation e test valide, includere un "buffer" di `numLags` campioni:
```matlab
% Dopo splitTrainValTest, includere il contesto dal training
% Per validation:
validationWithContext = [training(end-numLags+1:end, :); validation];
[~, validationNorm, ~] = normalizeZScore(training, validationWithContext, columnsToNormalize);
[xValidation, yValidation, timeVectorValidation] = createLstmSequences(validationNorm, numLags, predictors, target);
% Poi filtrare per tenere solo le sequenze relative ai giorni di validation

% Stesso approccio per il test set
```

---

## 5. Verifiche con MATLAB Tools

### `checkcode` — Risultati
- `LSTM.m`: ✅ Nessun warning
- `createLstmSequences.m`: ✅ Nessun warning

> I bug identificati sono **logici/architetturali**, non sintattici — non rilevabili da `checkcode`.

---

## 6. Conclusioni

Il problema principale delle performance scarse è quasi certamente il **BUG #1**: le sequenze di input hanno shape `[48 × 5]` (time-major) invece del formato atteso da MATLAB `[5 × 48]` (feature-major). Questo fa sì che la rete impari a correlare le 5 feature come se fossero 5 time steps, ignorando completamente la struttura temporale dei 48 lag. **Questa singola correzione potrebbe migliorare drasticamente le performance.**

Le altre correzioni (architettura più semplice, early stopping, strategia di split migliorata) sono comunque raccomandate per ottenere risultati ottimali.
