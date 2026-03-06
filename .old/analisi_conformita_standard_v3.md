# Analisi di Conformità agli Standard di Codifica MATLAB — v3

> **Data analisi:** 2026-03-05  
> **Riferimento standard:** [matlab-coding-standards.instructions.md](../.github/instructions/matlab-coding-standards.instructions.md) (basato su [MathWorks MATLAB Coding Guidelines](https://github.com/mathworks/MATLAB-Coding-Guidelines), CC BY 4.0)  
> **File analizzati:** 10 file `.m` (1 script principale + 9 funzioni in `Scripts/`)  
> **Strumenti:** analisi manuale + `checkcode` MATLAB Code Analyzer

---

## 1. Riepilogo Esecutivo

| Categoria Standard | Stato Generale | Criticità |
|:---|:---:|:---|
| **Naming (Nomi)** | ⚠️ Parziale | Lingua italiana, `snake_case` invece di `lowerCamelCase` |
| **Stringhe** | ❌ Non conforme | `'apici singoli'` predominanti invece di `"doppi apici"` |
| **Dichiarazione di Funzione** | ✅ Buono | Struttura corretta, output limitati, `end` sempre presente |
| **Documentazione (H1 / help)** | ⚠️ Parziale | Presente in alcune funzioni, mancante o incompleta in altre |
| **Controllo Flusso** | ✅ Buono | Nidificazione contenuta, uso corretto di `break` |
| **Validazione Input** | ⚠️ Parziale | Usa `assert`/`if-error`; manca il blocco `arguments` moderno |
| **Percorsi File** | ⚠️ Misto | `fullfile` usato correttamente in molti punti; eccezione in `LSTM.m` |
| **Struct e Dati** | ⚠️ Parziale | Campi aggiunti incrementalmente invece che in blocco unico |
| **Funzioni Deprecate** | ❌ Non conforme | `datestr`/`now` usati in `LSTM.m` e `save_models.m` |
| **Code Analyzer** | ⚠️ Warning presenti | 12 warning totali da correggere prima del commit |

**Punteggio complessivo stimato: 6.5 / 10**

---

## 2. Analisi per File

---

### 2.1 `LSTM.m` — Script Principale

**Tipo:** Script  
**Dimensione:** 152 righe  
**Code Analyzer:** 9 warning

#### ✅ Conformità

- Utilizzo di `%%` per separare le sezioni logiche (regola §commenti: sezioni)
- Utilizzo di `Name=Value` syntax per `trainingOptions` (R2021a+, regola §function-calls)
- Utilizzo di `fullfile` per il path del file `.mat` (riga 151)
- Nessun `global` o `persistent` improprio
- Uso di `~` per ignorare output inutilizzati (regola §syntax)

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 8 | §syntax – percorsi | `addpath(root + "\\Scripts")` — usa operatore `+` per path | `addpath(fullfile(root, "Scripts"))` |
| 27 | §stringhe | `predittori` è un cell array di `'apici singoli'` invece di `"doppi apici"` | `predittori = {"AAC_energy", ...}` |
| 50–51 | §variabili | `params_norm` in riga 50 è mai più usata (sovrascritta a riga 51) — assegnazione inutile | Usare `~` per il terzo output di riga 50 |
| 57–116 | §struct | La struct `net_struct` viene popolata campo per campo. | Definirla in un unico blocco con `struct(...)` |
| 103, 150 | **Deprecata** | `datestr(now, ...)` — funzione deprecata | `string(datetime("now", "Format", "yyyy_MM_dd_HH_mm_ss"))` |
| 143, 145, 147 | §formatting | Mancano i `;` a fine riga (output stampato a schermo in modo inatteso) | Aggiungere `;` |
| Globale | §naming – lingua | Variabili e commenti in italiano (`predittori`, `giorniTest`, ecc.) | Preferire inglese per portabilità |

---

### 2.2 `load_datas.m`

**Tipo:** Funzione  
**Dimensione:** 41 righe  
**Code Analyzer:** 8 warning

#### ✅ Conformità

- H1 line presente e ben strutturata con sezioni `INPUT` e `OUTPUT`
- `end` presente
- `fullfile` usato correttamente per la costruzione dei path
- `switch-otherwise` con blocco `otherwise` (regola §control-flow)
- Uso di `error()` nel blocco `otherwise` con messaggio descrittivo

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 15,18,21,24 | §load – conflitti | `load(file)` senza specificare variabili: rischio shadowing | `load(file, 'AAC_energy', 'time_vector')` |
| 30,31 | §load – conflitti | `load(...)` senza elenco variabili (warning Code Analyzer) | Specificare le variabili da caricare |
| 36,37 | Code Analyzer | Variabili `AAC_energy`, `time_vector` non inizializzate esplicitamente prima di `load` | Inizializzare a `[]` prima del `switch` |
| 1 | §naming – funzione | Nome `load_datas` (italiano, snake_case); regola: `lowerCamelCase` inglese | `loadZoneData` |
| 14,17,... | §stringhe | Stringhe come `'Zone_1016_Anagnina'` con apici singoli | Usare `"Zone_1016_Anagnina"` |
| — | §arguments | Nessun blocco `arguments` per validare `root` e `zone_id` | Aggiungere blocco `arguments` |

---

### 2.3 `normalize_zscore.m`

**Tipo:** Funzione  
**Dimensione:** 116 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning dal Code Analyzer
- Documentazione H1 presente
- Pre-allocazione degli output prima del loop (regola §control-flow)
- Gestione dei NaN con `'omitnan'`
- Uso di `matlab.lang.makeValidName` per nomi di campo sicuri
- `end` finale presente
- Logica corretta di normalizzazione (solo parametri del training applicati al test)

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | Nome italiano+snake_case (`colonne_da_normalizzare`) | `normalize_zscore(trainSet, testSet, columnNames)` |
| 13–38 | §arguments | Validazione tramite `assert`: stile datato | Sostituire con blocco `arguments` |
| 24 | §stringhe | `iscellstr` deprecato in favore di `iscell` + `cellfun(@ischar,...)` | Aggiornare o usare `string array` |
| 60,86 | §formatting | `if sg_is_zero, sg = ...; end` su una riga sola (più statement per riga) | Espandere su più righe |
| 114 | §stringhe | Stringa con apici singoli `'Normalizzate tabelle...'` | Usare doppi apici `"..."` |

---

### 2.4 `create_lags_LSTMNN.m`

**Tipo:** Funzione  
**Dimensione:** 87 righe  
**Code Analyzer:** 1 warning (bug critico)

#### ✅ Conformità

- Pre-allocazione di `XTrain_temp`, `YTrain_temp`, `time_temp` prima del loop
- Uso di `NaT` per pre-allocare vettori datetime
- `end` finale presente

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| **29** | **§control-flow — BUG** | Blocchi `if` e `else` identici: condizione senza effetto | Rimuovere il ramo `if/else`, tenere solo `Ymat = table2array(tbl(:, target))` |
| 1 | §naming | Nome funzione in `snake_case` misto (`create_lags_LSTMNN`) | `createSequencesLstm` |
| 3–8 | §arguments | Validazione con `assert`/`if-error` invece di blocco `arguments` | Aggiungere blocco `arguments` |
| Globale | §naming – lingua | Variabili in italiano (`giorni_unici`, `idx_sequenza`, ecc.) | Preferire inglese |
| 13,23,39–41 | §stringhe | `disp(['...'])` con apici singoli | Usare `fprintf` o `disp("...")` |

---

### 2.5 `create_lags_regression_learner.m`

**Tipo:** Funzione  
**Dimensione:** 89 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning dal Code Analyzer
- Pre-allocazione di `dati_matrix` con `NaN` e `time_vector_out` con `NaT`
- `end` presente
- Buona struttura in loop nidificato con variabili chiaramente nominate

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | Nome lungo in snake_case (`create_lags_regression_learner`) | `createLagsRegressionLearner` |
| 3 | §style | `tabella_input = dati_input` — copia inutile della variabile di input | Rimuovere, usare `dati_input` direttamente |
| 6–10 | §arguments | Validazione manuale con `assert`/`if-error` | Aggiungere blocco `arguments` |
| 15 | §stringhe | Mix: `'Giorni totali: ' + string(...)` — stile inconsistente | Usare `fprintf("Giorni totali: %d\n", ...)` |
| Globale | §naming – lingua | Tutte le variabili in italiano | Preferire inglese |

---

### 2.6 `normalize_zscore_2.m`

**Tipo:** Funzione  
**Dimensione:** 74 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning Code Analyzer
- H1 line e documentazione present
- Pre-allocazione tramite copia di `datas`
- `end` presente
- Uso corretto di `matlab.lang.makeValidName`

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | Nome italiano+snake_case (`normalize_zscore_2`) — il suffisso `_2` non è descrittivo | `normalizeZscoreFull` o `normalizeData` |
| 15–28 | §arguments | Validazione manuale con `assert` | Aggiungere blocco `arguments` |
| 50, 55, 59 | §formatting | Statements multipli su righe singole con virgola (es. `if sg_is_zero, sg = ...; end`) | Espandere su più righe |
| 73 | §stringhe | `disp('Normalizzazione Z-score...')` con apici singoli | Usare `"..."` |

---

### 2.7 `get_training_test_validation.m`

**Tipo:** Funzione  
**Dimensione:** 25 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning Code Analyzer
- Funzione compatta e ben leggibile
- `end` presente

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | `get_training_test_validation` — snake_case, nome molto lungo (>32 char: 34 char) | `splitTrainValTest` |
| 4–8 | §arguments | Nessun blocco `arguments` | Aggiungere blocco `arguments` |
| Globale | §naming – lingua | Commenti e variabili in italiano | Preferire inglese |
| 23 | §stringhe | `disp('...')` con apici singoli | Usare `"..."` |

---

### 2.8 `get_training_test3.m`

**Tipo:** Funzione  
**Dimensione:** 21 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning Code Analyzer
- Funzione semplice e leggibile
- `end` presente

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | `get_training_test3` — snake_case + suffisso numerico opaco | `splitTrainTest` |
| 1 | §naming – lunghezza | Nome >32 char se si include il percorso; borderline ma accettabile (\<= 32) | — |
| 4–8 | §arguments | Nessun blocco `arguments` | Aggiungere blocco `arguments` |
| Globale | §naming – lingua | Variabili e commenti in italiano | Preferire inglese |

---

### 2.9 `get_best_indicators.m`

**Tipo:** Funzione  
**Dimensione:** 20 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning Code Analyzer
- Logica chiara con `find(..., 1, 'first')`
- `end` presente
- Output unico, funzione compatta

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | `get_best_indicators` — snake_case | `getBestIndicators` |
| 2–3 | §doc | Nessuna riga H1 immediatamente dopo la dichiarazione | Aggiungere `% getBestIndicators ...` come H1 |
| — | §arguments | Nessun blocco `arguments` per validare `info` | Aggiungere `arguments; info; end` |
| 3 | §style | `info_net = info` — copia inutile | Rimuovere, usare `info` direttamente |

---

### 2.10 `get_indicators.m`

**Tipo:** Funzione  
**Dimensione:** 20 righe  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Nessun warning Code Analyzer
- Formule matematiche chiare e concise
- `end` presente

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| 1 | §naming | `get_indicators` — snake_case | `computeMetrics` o `getIndicators` (lowerCamelCase) |
| 2 | §doc | Nessuna riga H1 dopo la dichiarazione | Aggiungere H1 line |
| — | §arguments | Nessun blocco `arguments` | Aggiungere blocco `arguments` |
| 3–5 | §comments | Codice commentato lasciato nel file | Rimuovere o spostare in un branch |

---

### 2.11 `get_plot.m` e `get_plot_RL.m`

**Tipo:** Funzioni di visualizzazione  
**Code Analyzer:** ✅ Nessun warning

#### ✅ Conformità

- Documentazione H1 presente in entrambe
- `end` presente
- Struttura grafica chiara e leggibile

#### ⚠️ Non Conformità

| File | Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|:---|
| `get_plot_RL.m` | 1 | §naming – lunghezza | Nome >32 char: `figure_regression_learner` (27 char output, ok) — ma il nome funzione `get_plot_RL` è snake_case | `getPlotRl` o `plotResults` |
| Entrambi | varie | §stringhe | Uso di apici singoli in `legend('...')`, `title('...')`, `xlabel('...')` | Usare doppi apici `"..."` |
| Entrambi | — | §arguments | Nessun blocco `arguments` per validare gli input | Aggiungere blocco `arguments` |
| `get_plot.m` | 2–3 | §doc | H1 descrive parametri rimossi (`net`, `Xnorm`) — documentazione obsoleta | Aggiornare la documentazione |

---

### 2.12 `save_models.m`

**Tipo:** Funzione  
**Dimensione:** 44 righe  
**Code Analyzer:** 2 warning

#### ✅ Conformità

- H1 line e documentazione ben strutturata
- `fullfile` usato correttamente
- Gestione del caso file non esistente con `struct()` vuoto
- `end` presente
- Salvataggio in formato `-v7.3` per dataset grandi

#### ⚠️ Non Conformità

| Riga | Regola Violata | Descrizione | Correzione suggerita |
|:---|:---|:---|:---|
| **10** | **Deprecata** | `datestr(now, 'yyyy_mm_dd')` | `string(datetime("now", "Format", "yyyy_MM_dd"))` |
| 1 | §naming | `save_models` — snake_case | `saveModels` |
| 10 | §stringhe | Formato data con apici singoli `'yyyy_mm_dd'` | Usare doppi apici |
| — | §arguments | Nessun blocco `arguments` | Aggiungere blocco `arguments` |

---

## 3. Riepilogo Warning Code Analyzer per File

| File | N° Warning | Warning principali |
|:---|:---:|:---|
| `LSTM.m` | **9** | `datestr`/`now` deprecati; variabili inutilizzate (riga 50); mancano `;` |
| `load_datas.m` | **8** | `load` senza variabili specificate; variabili non inizializzate |
| `create_lags_LSTMNN.m` | **1** | **BUG:** rami `if/else` identici (riga 29) |
| `save_models.m` | **2** | `datestr`/`now` deprecati |
| Tutti gli altri | 0 | — |
| **TOTALE** | **20** | |

> [!CAUTION]
> Il warning a riga 29 di `create_lags_LSTMNN.m` segnala un **bug nel codice**: i due rami del `if/else` sono identici. La condizione non ha alcun effetto. Questo va corretto prima di qualsiasi utilizzo in produzione.

> [!WARNING]
> La regola §code-quality richiede che **tutti** i warning del Code Analyzer vengano corretti prima del commit. Attualmente sono presenti **20 warning** totali.

---

## 4. Conformità per Categoria Standard

### §1 Naming (Nomi)

| Regola | Conformità | Note |
|:---|:---:|:---|
| Nomi in lingua comune (inglese) | ❌ | Tutti i file usano italiano per variabili, commenti, messaggi |
| Lunghezza ≤ 32 char | ✅ | Nessun nome supera il limite |
| `lowerCamelCase` per funzioni | ❌ | Tutte le funzioni usano `snake_case` |
| Verbo/frase verbale per funzioni | ⚠️ | Molte usano `get_` (accettabile) ma non sempre un verbo chiaro |
| `UpperCamelCase` per Name-Value | ✅ | Usato correttamente in `trainingOptions` |
| Evitare nomi negativi (`isNot`) | ✅ | Rispettato |

### §2 Statements e Espressioni

| Regola | Conformità | Note |
|:---|:---:|:---|
| Un solo statement per riga | ⚠️ | Violato in più `if sg_is_zero, ...; end` single-line |
| Evitare `global`/`persistent` | ✅ | Non usati |
| `fullfile` per path | ⚠️ | Usato ovunque tranne `LSTM.m` riga 8 |
| No `eval` / `evalin` / `assignin` | ✅ | Non usati |
| `~` per output inutilizzati | ✅ | Usato |
| `Name=Value` syntax (R2021a) | ✅ | Usato in `trainingOptions` |
| Switch con `otherwise` | ✅ | Rispettato in `load_datas.m` |
| Nidificazione ≤ 5 livelli | ✅ | Max 3 livelli osservati |

### §3 Formatting

| Regola | Conformità | Note |
|:---|:---:|:---|
| 4 spazi per indentazione | ✅ | Rispettato globalmente |
| Righe ≤ 120 caratteri | ⚠️ | `get_plot_RL.m` riga 11: ~125 char |
| Nessun spazio dopo `(`, `[`, `{` | ✅ | Rispettato |
| Spazi attorno a `=` in assegnazione | ✅ | Rispettato |
| Sezioni con `%%` | ✅ | Usate bene in `LSTM.m` |

### §4 Commenti

| Regola | Conformità | Note |
|:---|:---:|:---|
| Almeno uno spazio dopo `%` | ✅ | Rispettato |
| H1 line subito dopo la dichiarazione | ⚠️ | Mancante in `get_indicators.m`, `get_best_indicators.m` |
| Commento prima del codice spiegato | ✅ | Generalmente rispettato |
| Codice commentato non lasciato nel file | ⚠️ | `get_plot.m` (righe 3–5), `get_indicators.m` (righe 3–5) |

### §5 Funzioni

| Regola | Conformità | Note |
|:---|:---:|:---|
| Nome file = nome funzione | ✅ | Rispettato ovunque |
| `end` alla fine di ogni funzione | ✅ | Rispettato |
| Max 6 argomenti input | ✅ | Max 5 osservato (`get_plot_RL`) |
| Max 4 argomenti output | ✅ | Max 3 osservato (`normalize_zscore`) |
| Blocco `arguments` per validazione | ❌ | Nessuna funzione lo utilizza |
| No ripetizione di blocchi di codice | ⚠️ | `normalize_zscore` e `normalize_zscore_2` condividono logica duplicata |

### §6 Gestione Errori

| Regola | Conformità | Note |
|:---|:---:|:---|
| Fix tutti i warning Code Analyzer | ❌ | 20 warning attivi |
| Messaggi di errore descrittivi | ✅ | I messaggi `error(...)` sono chiari |
| No `try-catch` per flow normale | ✅ | Non usato in modo improprio |
| No `throwAsCaller` | ✅ | Non usato |

### §7 Stringhe

| Regola | Conformità | Note |
|:---|:---:|:---|
| Usare `"` invece di `'` per stringhe | ❌ | Predomina l'uso di `'apici singoli'` in quasi tutti i file |

---

## 5. Priorità di Intervento

### 🔴 Alta Priorità (impatto correttezza o qualità professionale)

1. **[BUG] `create_lags_LSTMNN.m` riga 29** — Rami `if/else` identici: la condizione è inerte. Rimuovere la struttura `if/else` ridondante.
2. **`LSTM.m` + `save_models.m`** — Sostituire `datestr(now, ...)` con l'API moderna `datetime("now", ...)`.
3. **`load_datas.m`** — Specificare le variabili in ogni chiamata `load(...)` per evitare conflitti di shadowing.
4. **`LSTM.m` riga 50** — L'output `params_norm` è immediatamente sovrascritto a riga 51; usare `~` per il terzo output.
5. **`LSTM.m` righe 143, 145, 147** — Aggiungere `;` per evitare output indesiderato in console.

### 🟡 Media Priorità (stile e manutenibilità)

6. **Tutti i file** — Sostituire apici singoli `'...'` con doppi apici `"..."` per le stringhe di testo.
7. **Tutte le funzioni** — Aggiungere blocchi `arguments` per la validazione degli input (stile R2019b+).
8. **`LSTM.m` riga 8** — Sostituire `root + "\\Scripts"` con `fullfile(root, "Scripts")`.
9. **`normalize_zscore.m` e `normalize_zscore_2.m`** — Valutare refactoring per eliminare la duplicazione di logica.
10. **`get_best_indicators.m` riga 3** — Rimuovere la copia inutile `info_net = info`.

### 🟢 Bassa Priorità (convenzioni e adozione futura)

11. **Tutti i file** — Rinominare funzioni e variabili da `snake_case` italiano a `lowerCamelCase` inglese.
12. **`get_indicators.m`, `get_best_indicators.m`** — Aggiungere le righe H1 mancanti.
13. **`get_plot.m`** — Aggiornare la documentazione H1 (fa riferimento a parametri rimossi).
14. **Tutti `disp(...)` e `fprintf`** — Uniformare lo stile dei messaggi diagnostici.

---

## 6. Esempi di Refactoring Raccomandati

### 6.1 Blocco `arguments` (tutte le funzioni)

```matlab
% Prima (stile datato)
function out = load_datas(root, zone_id)
    assert(isstring(root) || ischar(root), 'root deve essere stringa');
    ...

% Dopo (stile moderno R2019b+)
function out = loadZoneData(root, zoneId)
    arguments
        root   (1,1) string
        zoneId (1,1) double {mustBePositive, mustBeInteger}
    end
    ...
```

### 6.2 Sostituzione `datestr(now, ...)` → `datetime`

```matlab
% Prima (deprecato)
currentDate = datestr(now, 'yyyy_mm_dd');

% Dopo (moderno)
currentDate = string(datetime("now", "Format", "yyyy_MM_dd"));
```

### 6.3 Stringhe: doppi apici

```matlab
% Prima
predittori = {'AAC_energy','precipprob','temp'};
disp(['Dati per la zona "' zone_full_name '" caricati.']);

% Dopo
predittori = {"AAC_energy", "precipprob", "temp"};
disp("Dati per la zona """ + zoneName + """ caricati.");
```

### 6.4 Percorsi: sempre `fullfile`

```matlab
% Prima (LSTM.m riga 8)
addpath(root + "\\Scripts");

% Dopo
addpath(fullfile(root, "Scripts"));
```

### 6.5 Fix bug `create_lags_LSTMNN.m` riga 29

```matlab
% Prima (BUG: rami identici)
if ischar(target) || (isstring(target) && isscalar(target))
    Ymat = table2array(tbl(:, target));
else
    Ymat = table2array(tbl(:, target));
end

% Dopo
Ymat = table2array(tbl(:, target));
```

---

*Documento generato automaticamente tramite analisi del repository e Code Analyzer MATLAB.*
