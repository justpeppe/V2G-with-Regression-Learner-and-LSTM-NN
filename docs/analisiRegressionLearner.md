# Analisi Tecnica Completa del Progetto: V2G con Modelli Classici (Regression Learner Ensemble)

**Autore:** Analisi generata da Antigravity AI  
**Data:** Marzo 2026  
**Progetto:** V2G-with-Regression-Learner-and-LSTM-NN  

---

## Indice

1. [Contesto Applicativo: Vehicle-to-Grid (V2G) (Partendo da Zero)](#1-contesto-applicativo-vehicle-to-grid-v2g-partendo-da-zero)
2. [Il Problema: Previsione del Consumo Energetico](#2-il-problema-previsione-del-consumo-energetico)
3. [Paradigma di Risoluzione: Da Sequenze a Dati Tabulari](#3-paradigma-di-risoluzione-da-sequenze-a-dati-tabulari)
4. [La Cassetta degli Attrezzi: I Modelli del Regression Learner](#4-la-cassetta-degli-attrezzi-i-modelli-del-regression-learner)
5. [Variabili Iniziali: Il Cuore della Configurazione](#5-variabili-iniziali-il-cuore-della-configurazione)
6. [Architettura del Codice e Flusso di Esecuzione](#6-architettura-del-codice-e-flusso-di-esecuzione)
7. [Struttura e Costruzione dei Dati: Dal Grezzo ai Lag](#7-struttura-e-costruzione-dei-dati-dal-grezzo-ai-lag)
8. [Analisi Dettagliata degli Script](#8-analisi-dettagliata-degli-script)
9. [Previsione e Generalizzazione Cross-Zona](#9-previsione-e-generalizzazione-cross-zona)
10. [Le Figure Generated: Capire e Leggere i Visual](#10-le-figure-generated-capire-e-leggere-i-visual)
11. [I Log Salvati: Struttura e Interpretazione](#11-i-log-salvati-struttura-e-interpretazione)
12. [Analisi Critica, Limitazioni e Confronto con LSTM](#12-analisi-critica-limitazioni-e-confronto-con-lstm)

---

## 1. Contesto Applicativo: Vehicle-to-Grid (V2G) (Partendo da Zero)

### Cos'è il V2G?
La transizione globale verso fonti di energia rinnovabile e l'elettrificazione del settore dei trasporti hanno portato all'emergere di un paradigma rivoluzionario: il **Vehicle-to-Grid (V2G)**. In un sistema energetico tradizionale, il flusso di energia è unidirezionale: dalla centrale di produzione alla casa o all'automobile. L'automobile elettrica (EV) viene vista esclusivamente come un "carico", ovvero un dispositivo che assorbe energia e stressa la rete, specialmente quando milioni di veicoli si collegano simultaneamente (il cosiddetto "ricarica serale").

Il concetto di V2G ribalta questa prospettiva. Un veicolo elettrico è essenzialmente una massiccia batteria su ruote. Quando è parcheggiato (il che statisticamente avviene per oltre il 90% del suo ciclo di vita), l'EV può funzionare non solo come recettore di energia ("Grid-to-Vehicle" o G2V), ma anche come **fornitore di energia** verso la rete elettrica. 

I vantaggi di questo sistema bidirezionale sono ciclopici:
1. **Peak Shaving (Taglio dei picchi):** Durante le ore di massima richiesta energetica (es. 19:00 - 21:00), quando le centrali a gas devono essere accesse per far fronte al consumo, le auto attaccate alla presa possono cedere parte della loro carica per abbassare il fabbisogno della rete, evitando sovraccarichi.
2. **Load Leveling (Livellamento del carico):** L'energia immagazzinata nei veicoli può mitigare la curva di "Duck Curve", tipica dei sistemi ad alta penetrazione solare in cui al tramonto c'è un crollo di produzione e un'impennata di domanda. Le auto colmano quel vuoto energetico.
3. **Arbitraggio Energetico:** Il proprietario del veicolo si trasforma in un "prosumer" (produttore + consumatore). Può acquistare energia di notte quando il costo al kWh è bassissimo o negativo (a causa dell'eccesso di produzione eolica/idroelettrica) e rivenderla alla rete durante il giorno quando il prezzo esplode.
4. **Regolazione della Frequenza:** La rete elettrica deve mantenere una frequenza stabile (50 Hz in Europa). Iniezioni o prelievi istantanei dai veicoli collegati aiutano a stabilizzare la frequenza meglio di qualsiasi centrale meccanica a causa della velocità di risposta dell'elettronica di potenza e degli inverter.

### L'Importanza della Previsione del Consumo
Perché il V2G sia fattibile economicamente ed operativamente, non basta "avere batterie". È necessario un cervello centrale (Energy Management System - EMS) o un aggregatore che prenda decisioni ottimali in tempo reale: *Quando carico? Quando scarico? Quanta energia lascio all'utente per viaggiare l'indomani?*

Queste decisioni dipendono in maniera cruciale da **cosa succederà nel prossimo futuro**. Se l'aggregatore non ha idea di quale sarà il consumo energetico di un dato quartiere o distretto (una Zona), non può sapere se ci sarà bisogno di iniettare energia (per sopperire a un picco) o assorbirla (per evitare sovratensioni locali). 

Pertanto, un modello di previsione accurato del consumo elettrico (Load Forecasting) è l'enzima fondamentale per le logiche del V2G. Un errore di previsione comporta due gravi rischi:
- **Sottostima del consumo:** La rete si aspetta un consumo basso, le auto vengono messe in carica, si verifica un picco inatteso e salta il trasformatore locale.
- **Sovrastima del consumo:** Le auto vengono scaricate per supportare un picco inesistente; l'utente si ritroverà l'auto scarica al mattino per andare a lavoro e l'aggregatore avrà sprecato cicli di vita vitale (degrado) della batteria.

### L'Ambito di Progetto
Il progetto si concentra sul prevendere in maniera quantitativa (kWh) il fabbisogno energetico aggregato di porzioni urbane precise della città di **Roma, anno 2023**, utilizzando un approccio di Machine Learning. Le porzioni urbane sono state suddivise in "Zone" codificate numericamente (Zona 8: Anagnina, Zona 9: Trieste, Zona 10 e 11: Tor di Quinto e aree limitrofe). I dati sono ad elevata frequenza: risoluzione oraria di **30 minuti** (half-hourly).

---

## 2. Il Problema: Previsione del Consumo Energetico

### Analisi del Problema
Prevedere il consumo è un problema matematico inquadrabile come **regressione su serie temporali** (Time-Series Regression). Stiamo cercando di prevedere una variabile continua target $y$ (consumo elettrico in kWh al tempo $t$, ovvero `AAC_energy`) basandoci su informazioni storiche e descrittive fino al momento $t-1$.

Il consumo di energia non è un fenomeno casuale (white noise), ma mostra forti componenti sistemiche:
- **Stagionalità Giornaliera:** Ciclo continuo notte-giorno. Picchi mattutini al risveglio, picchi serali al rientro lavorativo, valli notturne profonde.
- **Stagionalità Settimanale:** I giorni feriali (lunedì-venerdì) sono molto diversi dai weekend (sabato-domenica).
- **Auto-correlazione forte (Inerzia Termica ed Elettrica):** Il consumo al tempo $t$ è fortemente correlato al consumo al tempo $t-1$ o $t-48$ (lo stesso momento del giorno precedente).
- **Fattori Esogeni (Meteo):** Se l'aria condizionata è accesa (temperatura alta d'estate) o se accendono pompe di calore (bassa d'inverno), il consumo schizza. Anche fenomeni come la precipitazione inducono le persone a stare a casa, aumentando il consumo domestico.

### Dataset a Disposizione ed Estratti
Per catturare le suddette dinamiche si possiedono diversi flussi di informazione (Predittori/Features) fusi assieme:
1. **Dati Autoregressivi / Endogeni:** `AAC_energy`, il consumo. Si utilizza la storia del consumo stesso per capire il trend attuale.
2. **Dati Meteorologici:**
   - `temp`: Temperatura media in gradi Celsius.
   - `windspeed`: Velocità del vento. Un maggiore vento comporta un maggiore raffreddamento percettivo che può influenzare il riscaldamento, o può essere proxy di passaggi di perturbazioni importanti.
   - `precipprob`: Probabilità di precipitazione in %. Proxy per le condizioni di luce naturale o confinamento delle persone al chiuso.
3. **Dati di Calendario:** 
   - `holiday_indicator`: Boleano 0/1, derivato caricando un calendario italiano di festività (`holidays.mat`). È essenziale poiché un "Lunedì di Pasquetta" non si comporta come un lunedì lavorativo.
4. **Dati Tempo-Ciclico:** 
   - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`. Il tempo è circolare. Il mese e le ore del giorno non sono rette ma curve chiuse (un orologio). Rappresentando l'ora "23:30" come 23.5 e l'ora "00:00" come 0, i numeri sono molto distanti, ma fisicamente c'è una continuità di 30 minuti. Attraverso seno e coseno l'orario viene "avvolto" su un cerchio, per permettere ad algoritmi lineari di decodificarne la continuità locale: $sin(2\pi \cdot hour / 24)$ e $cos(2\pi \cdot hour / 24)$.

La natura dei dati temporali però è frammentata: si osserva il progetto organizzato tramite "Cluster" di dati disponibili distaccati fra loro temporalmente. Ci sono salti di giorni o settimane (e.g., metà febbraio per 14 giorni, poi nulla fino a fine maggio). La mancanza di un anno contiguo rende problematica l'estrapolazione della macro-stagionalità annuale: il modello dovrà concentrarsi molto di più sugli effetti a breve termine e sulle variabili metereologiche che gli offrono indizi contestuali.

---

## 3. Paradigma di Risoluzione: Da Sequenze a Dati Tabulari

Mentre una rete LSTM ("Long Short-Term Memory", analizzata in `LSTM.m`) fa scorrere le sequenze temporali nativamente gestendo l'informazione del "passato" incapsulata nello "stato nascosto" (hidden state) ad ogni passaggio iterativo, i modelli di **Regression Learner classici** non hanno concetto di 'tempo' e 'stato'. Sono **modelli memory-less**, puramente tabulari. 

### Il Flatting del Contesto e la "Lag Matrix"
Come fa un algoritmo Random Forest o una SVM a capire che stiamo trattando variazioni temporali? Il segreto, e la differenza macroscopica e strutturale tra il file `LSTM.m` e `RegressionLearner.m`, è **la finestra "Finestratura Sliding" e il Flattening**. 

Invece di inviare a un modello 9 feature su un pacchetto che scorre nel tempo, si dice al modello:
*"Guarda, io voglio che tu preveda il consumo a t=0. Per farlo ti do: "*
- *"La temperatura a t-1, il vento a t-1, l'orario a t-1"*
- *"La temperatura a t-2, il vento a t-2, l'orario a t-2"*
- ...
- *"La temperatura a t-48, il vento a t-48, l'orario a t-48"*

Avendo definito `numLags = 48` (che corrispondono esattamente a 24 ore a 30 min per step), se passiamo al modello 9 *features* per ogni istante passato, la rete non prende una matrice (48x9), ma prende la matrice srotolandola e stirandola orizzontalmente fino a creare un unico lunghissimo vettore spaziale largo **432 colonne (9 feature x 48 step + la target come label successiva)**. 

Quindi i modelli utilizzati si troveranno davanti dataset Larghi anziché Profondi. In `RegressionLearner.m`, il passaggio da tempo a puro dato tabulare "largo" avviene grazie alla fondamentale funzione custom: `createRegressionLags.m`.

Questo passaggio è un'arma a doppio taglio:
- **Vantaggi:** Si sblocca la possibilità di usufruire di innumerevoli e stabili algoritmi tabulari (Ensemble Tree-based, SVM) incredibilmente efficienti nell'imparare interazioni complesse senza soffrire del problema di convergenza o instabilità tipico del Deep Learning. Inoltre l'interpretabilità può essere maggiore.
- **Svantaggi:** Ignora la struttura sequenziale dei dati. La "Temperatura a t-47" e "Temperatura a t-48" per il Random Forest sono variabili viste in maniera diametralmente staccata e ortogonale, identicamente ignorate che abbiano un ordine logico o meno. Spetta all'albero di decisione fare innumerevoli tagli (splits) per ricostruire da capo le relazioni temporali nascoste nella larghezza immensa della tabella risultante. 

---

## 4. La Cassetta degli Attrezzi: I Modelli del Regression Learner

A differenza dell'`LSTM.m` che fonda le sue basi su librerie Deep Learning, all'interno del file `RegressionLearner.m` c'è un blocco selettivo che permette il training di paradigmi completamente diversi tra loro ma implementati tramite pacchetto "Regression Learner" automatizzato di MATLAB (Le librerie Machine Learning/Statistiche native). Questa implementazione "modulare" e switchabile è potentissima per fare un benchmark onesto di "Cosa funziona meglio?".

Il cuore di questa scelta è governata dalla variabile costante `modelType` ed instradata da uno `switch`:

1. **"ensemble" (LSBoost - Gradient Boosted Trees):** Un metodo formidabile. Il Gradient Boosting non crea un singolo grande "Albero dei Consumi", che inevitabilmente andrebbe a fare *overfitting* sui giorni in memoria. Costruisce una foresta di minialberi sequentially. Ogni piccolo regression tree apprende non sul target grezzo originale, ma sul "residuo", ovvero l'errore commesso dagli alberi che l'hanno preceduto, calcolato col Metodo dei Minimi Quadrati (Least Squares = LS). Questo "aggiusta la mira" ricorsivamente.
    *   **Impostazioni usate nel codice:** `NumLearningCycles: 300` (Costruisce 300 microalberi), `LearnRate: 0.05` (Evita che ogni albero "strappi" la decisione in modo aggressivo ma la muova col delicato tasso del 5%), `Learners: templateTree(MaxNumSplits, 6)` (Alberi tenuti super piccoli - i famosi stub - per cedere tutta la forza al collettivo dell'ensembe e limitare la complessita dei parametri memorizzati).
    *   **Ideale per:** Dati tabulari larghi (le famose 432 colonne prodotte dei 48 lag), gestione di outliers ed estrazione automatica dell'importanza feature.

2. **"tree" (Regression Tree Semplice - `fitrtree`):** Il cugino semplice dell'Ensemble. Costruisce un unico albero profondo e ramificato. Iperparetrizzato via `MaxNumSplits: 50` (Tantissimi nodi, cerca relazioni asticello profonde) e un `MinLeafSize: 5` (almeno 5 esempi in ogni foglia terminale per resistere ai rumori del microminuto).
    *   **Svantaggio:** Rischia moltissimo di memorizzare pattern casuali (overfitting) o di mancare variazioni macro non lineari.

3. **"svm" (Support Vector Machine via `fitrsvm`):** Algoritmo nato non per predire consumi (valori reali, non discreti/classificazioni) e adattato storicamente al tempo (SVR, Support Vector Regression). Impiega un kernel `rbf` (Radial Basis Function). L'SVM spinge le nostre 432 dimensioni "features srotolate" dentro uno spazio matematico infinito tramite il trucco kernel (kernel-trick) per tracciare il consumo lineare nell'iperspazio con tolleranza d'errore.
    *   **Pro:** Quando ci s'incappa in problemi complessi, lo spazio kernel srotola benissimo.
    *   **Contro:** Soffre tragicamente se non si normalizzano i dati perfettamente, con complessità `O(n^2)` per tempi di training che tendono ad esplodere se ci sono troppe righe create a causa dei vari Lag.

4. **"gpr" (Gaussian Process Regression - `fitrgp`):** L'approccio matematicamente e probabilisticamente più sofisticato. Costruisce una distribuzione di probabilità Bayesiana su infinite funzioni, adottando un kernel "Squared Exponential". Non fornisce solo un numero di consumo energetico previsto (ad es., previsione: "Ci saranno 1200 kW/h impiegati"), ma restituisce una certezza probabilistica legata all'evento e la deviazione standard del target, creando implicitamente limiti e confini fiduciari di incertezza (confidence bounds). 
    *   **Limitazioni applicative in questo progetto:** Non implementato per scale e rapidità. Le matricole GPR esploderebbero sui grandi numeri o sulle feature colossali, quindi spesso richiede pre-processing con PCA o riduzioni, oltre che ad impiegare un tempo non compatibile per addestramento veloce e sperimentale.

5. **"linear" (Lasso Regularization Regression):** È l'opposto e la nemesi di tutta l'infinita non polarizzazione e astrazione della serie LSTM. Modello elementare `y = ax + by + cZ`. Il solver utilizzato è con la limitazione ed impostazione di "Lasso Regularization". Questo impatta drammaticamente il numero di input (432 derivanti da `numLags*numFeatures` sono ridondanti e spesso nulli) portando progressivamente tutti i moltiplicatori di pattern "inutili o ripetitivi" al valore zero assoluto e perfetto.

---

## 5. Variabili Iniziali: Il Cuore della Configurazione

Le costanti definite a capo del main file `RegressionLearner.m` controllano e dettano tutta l'esecuzione strutturale del load forecasting:

*   **`numLags = 48`**: Il fulcro strategico del codice. Rappresenta l'ampiezza dell'orizzonte storico guardato prima di dare una previsione. Un `step` rappresenta 30 Minuti (half-hourly). Un `numLags` impostato alla modica cifra di 48 step significa che il Modello per indovinare il consumo elettrico ad Anagnina oggi alle ore 12:00 P.M. non darà conto delle 8 A.M. di oggi, ma si fisserà ossessivamente sulla storia da ora in poi e sulle variabili di ieri a fine ciclo, guardando tutte le 24 Ore temporali come una finestra storica che regredisce scorrendo con interazioni complesse sovrapposte. È il cosiddetto "Context Window".
*   **`useAutoregressive = true/false`**: Forse il toggle intellettualmente più profondo all'interno dell'esperimento che governa enormi ramificazioni di risultato in uscita.
    *   Se `true`, il sistema non si limiterà ad utilizzare agenti esterni / esogeni come vento, cicli temporali, sinusoidali, sole e nuvole (`exogenousPredictors`) ma incuterà una colonna aggiuntiva all'inferenza: *"Quanta energia ha incassato storicamente per step e per lag a quel punto?"*. Ovviamente sapere quanto inquinamento elettrico è avvenuto da ieri è il segreto indizio predominante che il Machine Learning attinge per non commettere catastrofici abbagli. 
    *   Se `false`, la rete è letteralmente blindata a non conoscere in anticipo lo storico elettrico in uscita passata e i carichi di consumo della stazione di rete. Può unire i puntini basandosi esclusivamente su cause atmosferiche esterne e "ritmo della giornata" e di festività in modo puramente esterno, sviscerando e stimolando i predittori "veri" causali rispetto al solo trascinamento temporale in inerzia passata. Un formidabile test empirico di causalità.
*   **`modelType`**: Variabile a catena vista a priori che dirige brutalmente su quale branca algoritmica statistica o di Machine Learning fuggire lo scaling, con opzioni come `ensemble`, `tree`, `svm`, `gpr`, `linear`.
*   **Formato Orario Dinamico:** Il naming del modello in salvataggio produce automaticamente identificativi di versione "Timestamp" ed appendibili su cartelle temporali `regEnsembleAutoReg[Ora]`. Automatismo prezioso per lo scorrimento progressivo in fase di sperimentazioni empiriche da parte del data scientist e del team o per le sessioni che si sommano.

---

## 6. Architettura del Codice e Flusso di Esecuzione

L'architettura in `RegressionLearner.m` si comporta e compie la computazione matematica nel seguente ordine algoritmico rigidissimo, progettato per evitare "Data Leakage" e produrre log testabili empiricamente di grandissima riproducibilità tecnica:

1.  **Dichiarazione Root Path:** Appende `addpath(root + "\Scripts")` in modo imperativo caricando di fatto l'intero "cuore logico", ovvero il sottobosco di utility isolato dal programma chiamante.
2.  **Preparazione e Logging Intelligente:** Tramite `diary(fullfile(reportDir, "Analysis_Log.txt"));` comanda lo stream della shell Output in locale e lo serializza fisicamente su un file testuale a disco, salvando la cronistoria d'addestramento ed i print descritti tra epoche matematiche in un file "Analysis_01" testuale utile post-esecuzione. Formidabile architettura.
3.  **Selezione Strategica Giorni (Il Crux del Testing Deterministico):** Si sguinzaglia `selectRepresentativeDays(datas)`.
4.  **Generazione "Features Circolari":** Sinusoide/Coseno implementazione della periodicità fissa e costante come da orologio e giornate della settimana (vedi sezione dati ciclici tempo).
5.  **Divisione Treni (Split Train/Val/Test):** Si avvia formalizzazione su `splitTrainValTest(datas, validationDays, testDays);`. Scissione secca di giorni di addestramenti pesanti vs le singole ed estrapolate mini-finestre di check validation che intersecano tra un passo epocale all'altro e la super severa sezione estraniata che è "L'isolotto finale della verità su Test", invisibile persino al ciclo di calibrazione parametri per i check post addestramento.
6.  **Z-Score Normalization Asimmetrica:** Fondamentale. Media e Varianza estrapolati unicamente in `trainingNorm` ed imposti e scalati per Validation e Test Set col calcolo "Mu/Sigma" dell'addestramento, scartando i mini sbalzi estranei validando e mantenendo l'oggettività metrica di addestramento su range -3 e +3, fondamentale ed insindacabile per Regression/SVM (Lasso esploderebbe coi grandi divari di temperatura tra i 38 e vento ai 5 vs -10 etc e kW sui mila)
7.  **Sliding e Produzione Tabellare per ML:** Produzione e dislocazione `createRegressionLags`.
8.  **Modellazione Addestrativo (`switch modelType`)** Formazione e fit con estrapolazione della scatola magica o dell'albergo forestrale.
9.  **Predict Predittiva** La funzione inferisce al volo sui tre insiemi predittivi, ovvero sul `yTrainPred = predict(mdl, XTrain);yValPred...`. 
10. **Matematica delle Metriche (Score di Qualità)** Genera tutti gli score su `computeMetrics`. Assolve a salvataggio automatico del pacchetto in .mat su chiavette e disco rigido tramite `saveModels` creando serializzazione permanente ed invariabile (pescando e includendo perfino nel file .mat architettato i parametri `normParams` che servono come la mappa del tesoro della chiave inversa per riscalare alla base originale).
11. **Renderizzazione e Salvataggio Forme / Figure** Ricalcolo figure auto, denormalizzate, con plot per grafici, salvando le Figure in file `fig` MATLAB ispezionabili sia visualmente (.png ad altissima e densissima precisione) e manipolabili. Il codice è automatizzato per compiere ciò per ACF, Time Series e Breakdown per ora, analitica in log ed in fine disattivare i report chiudendo il giornale (`diary off;`).

---

## 7. Struttura e Costruzione dei Dati: Dal Grezzo ai Lag

Prima di entrare negli script, esploriamo cosa accade ai dati.

I dati grezzi partono come file `.mat` separati (es: consumi di Trieste, meteo generale, calendario festività). La fusione è operata da `loadZoneData.m` che allinea, con il tempo `.time_vector`, consumo energetico vs clima e temperatura vs giorni rossi da calendario. 

La trasformazione dal grezzo alle famose 432 colonne (lag) avviene nel passaggio più peculiare del Machine Learning su serie temporale tabularizzata. Questo trucco inibitivo della "non direzionalità logica del tempo in algoritmi base" richiede e comporta una decapitazione strutturata dei flussi giornalieri:
**Il rischio critico dei Lag (Gestito brillantemente da `createRegressionLags.m`):**
Essendo i dati sparpagliati a finestre temporali staccate ('Cluster temporali' estinti da Febbraio e passanti a Giugno, eccitanti il vuoto profondo centrale di Primavera scossa):
Cosa accade se passiamo a `t(Oggi)` `X(Vento a t-48 ore del giorno e blocco prima, e.g. Febbraio)` quando Oggi risulta in Maggio e ieri non è un blocco mappato in input dataset originario e mancano i flussi intermedi (Buchi Intercalari o Data Spaced)?. La macchina interpolando crederebbe che all'improvviso si sia abbattuto un vortice di -15 grandi termici rispetto a ieri e trarrebbe le famose e catastrofiche "Out of Pattern" allucinazioni o spikes predittive. 

Ecco perché nello script dei Lag è inclusa una sezione anti-data-leakage e anti-salto interrotta che esclude categoricamente sezionamenti su blocchi che non godono in contiguità i periodi storici dei t-X lags richiesti. La stringente validazione "Vede e Preleva" solamente blocchi consecutivi con interezze temporali continue accertabili, portando la quantità di training row drasticamente in basso, ed incrementando enormemente la "qualità e la sincerità" su scala ed importanza ed evitando buchi trappola di "time shift falsi".

---

## 8. Analisi Dettagliata degli Script

Esploriamo adesso come la magia informatica ed algoritmica fluisce sotto la console, entrando negli script vitali associati.

### A. `createRegressionLags.m`
(Il cuore dell'elaborazione Tabellare Classica Machine Learning Memoryless).

**Cosa Fa?**
Questa funzione prende i dati ordinati temporali di input (Time Series normale in sequenza Pandeana/Table, lunga X righe e stretta Y features), il target Y ed il famigerante ed enorme intervallo storico richiesto da scandagliare al setto `regressors` in iterazione (nel nostro config variabile, 48 steps). Lo script deve generare la Mega Tabella delle Features Includendo e nominando esplicitamente con stringhe descrittive (e.g. `temp_t_1`, `temp_t_48`, `windspeed_t_20` ... ecc).

**Funzionamento:**
1. Analizza i Giorni Eseguiti: Determina e spezzetta in pacchetti `daysArr = dateshift... "start", "day"`. Questo garantisce unicità di ogni data.
2. Controllore "Nessun Buco" o `validConsecutiveDays`: Analizza per ciclo giorno corrente t0 contro vettore di ieri in iterazione T-1. Sottrae matrici da giorno a giorno. Il calcolo se risulta un salto vuoto "Differenza = 1 GG, Procede True/Valido Consecutive", ma se i giorni di stop o stasi hanno provocato gap grossolano i vuotamente invalide salterà le sezioni ed argina disordini e sbalzi "Time shifting Fake ed Inadatti" (Blocco preesistente per cui preannunciavamo la complessità di dataset frammentari e l'ingegno logico della gestione di Cluster distaccati per stagioni!).
3. Loops d'Innesco per estrapolazione vettoriale "Tabelle in Coda Lag": Partendo da 48 iterazioni calanti in reverse `lag = regressors:-1:1` procede accatastando `[previousData; currentData]` al blocco e travasando per lo shift indicizzato alla formula astratta in posizionamento scalato array di matrice. Genera le stringhe "Label Variabili" tramite SprintF dinamicamente allineate (`"temp_t_48"` o `featureNames`). 
4. Converte e Sigilla: Tranciate e rimosse le eccedenze o ritardate code invalide di "t=Pre start senza storage buffer" il tutto finisce riavvitato a "Tabella Datatype di Array" a output pronto in Regression ed ingoiabile in pasto dai Gradient Booster preimpostati, unito al target colonna (L'ultimo vettore).

### B. `plotRegressionResults.m`
(Generatore Diagnostico Temporale Visuale ad Alta Risoluzione e Spaziato Flessibile).

**Cosa Fa?**
Uno script disegnatore custom creato e plasmato per combattere i rimasugli logici dell'essere spezzati su blocchi frammentari e gap giornalieri "Day-Hole Spaced" dei Cluster. Le classiche stampe "Time plot" unirebbero le code graficamente ed unirebbero il termine "Giorno di Gennaio" alla linea successiva ed unica scampata su "Settembre Spaced Testing Phase" e traccerebbero linee folli rezzando sul chart e rendendolo ininterpretabile. Il programmatore interviene tracciando e spaccando appositamente il tutto.

**Funzionamento:** 
1. `Denormalize`: Riceve Vettori Y predetti col collare a +1 / -1 normalizzati z-score asimettrici e re-integra applicando retro moltipliazione di std_deviation + medie aritmeticate salvate. Es: riposiziona `(-1.3)` su scala vera reale Kw/h -> `(450 KwH)`. Reale come il sensore la leggertebbe alle cabine elettriche!
2. Logica anti Gap a Step di Logica `Logical Index Array`: Ignora il Datetime continuo asse X vero, si spoglia ed attribuisce index secchi, 1;2;3 in vettore continuo. Costruisce confini su fine e giorno calcolando mediane d'asse tra un salto l'altro `mask = dayStamps.. ; idx=find..`. Ricostruisce artificialmente i Muri (Lines) e le spaccature settimanali verticali ed etichette dinamiche ad "Orario inclinato a 45 gradi" in corrispondenza del centramento intero in step, generando leggibilità e correndo al respiro visivo per interpretazione delle sezioni Test set isolate per il Machine learner e gli Analisti. Stampa con precisione rossa per Y(Predetto) contro griglia Nera Y_Raw(Verità Reale Assoluta).

---

## 9. Previsione e Generalizzazione Cross-Zona

### Il File `RegressionLearner_Predictions.m`
Questo script incapsula l'equivalente della prova del nove statistica ed empirica di intelligenza estraepolativa che stacca formalmente questo Progetto Universale, testando non una zona di comfort ma proiettando come si comporterebbe al buio della cabina di Roma su altri confini (Cross Validation per Generalizzazione Area).

Mentre il modello e lo switch RegressionLearner base allena, traccia grafici di rientro ed emette file salvataggi nel vuoto, la fase `Predictions` in questione: The Reaper (il Mietitore dei Salvataggi e Giudice):
1.   Autodetermina la rotta automatica sul direttorio "Sessioni" ed estrae dinamicamente e temporaneamente l'elenco della directory per data alla pesca ed incapsulo magico automatico dell'ultimo "Mega Modello Salvato" recente in lista per test automatizzato. (No hard-coding sui permesi!).
2.   Identifica automaticamente il blocco `targetZones = [8, 9, 10, 11];` da spazzolare e ciclare in modo inescrutabile alla ricerca di una predizione totale per singola istanza in un loop immenso.
3. Lo script è dotato di varianze pre impostabili: Modalità Test su "Intero anno solare del file o `TestMode= ALL`", Modo su Rappresentativo ad estrapolazione o Custom. Ad ogni iterata di loop zona i dati vengono disallineati dalla loro fonte grezza, filtrati per Normalizzazione con le Variances del blocco del Modello Importata in Load(Il load dello split originario del Training Param!) calcolata internamente asimmetricalmente e prodotta ad allineamento con i "lagging `CreateRegressionLags` riapplicato pedissequamente contro test e target a medesimo parametro 48 configurato nell'origine dal genitore!

### Classificatore Vocale e Logica "Quality Automatic Analysis"
Un aspetto interessante di questo script è l'inserimento non solo di freddi numeri (RMSE, MAPE ecc) ma dell'implementazione "Analista Vocale" nel print da terminal, che trasforma logiche computazionali oscure in responsi verbali ad hoc (Esperti preprogrammati nel System Code e Feedback): 

- R2 sopra il "Santo Graal" dell' 80%? Il print automatico interviene sigillando un esito *"EXCELLENT, superiore al 0.80"* a beneficio e spunto visivo della ricerca, o retroscende alla classificazione formale: *"ACCEPTABLE Room for improvement" (0.50)".* Lo stesso fa col persistente, la soglia del Modello "Persistenza di Dummy Baseline". 
Se il LSBoost Model fallisce contro il persistere o lo ignora la collaudazione stamperà una diagnosi *"FAIL, BaseLine Fails against Persistence."*. Identico discorso per la deviazione oraria. Uno schema condizionale controlla l'orario di sfalsamento dei massimi error. Il computer interroga sé stesso: "Ha il massimo di errore nel set tra le 8 A.m. alle 12:00 mattutine?". Se la risposta è positiva il Feedback si trasforma nell' "Annotazione o Avviso": -> "Peak Error di Picco del mattino identificato. Causato e tipico in zone Commerciale/Uffici e Residential/Poli direzionali!". Il computer esegue autonomamente analisi critica su fasce critiche. Una intelligenza tecnica all'interno della predizione che va ammirata. 
La finalità cross è produrre i report massimi `Cross-Zone Summary` finali. Una tabella "Final Cross Zone" che unisce alla perfezione tutto in una visibile comparazione delle devianze zonali! (RMSE, ZONE, MAE, PERSISTANCE MAE...).

---

## 10. Le Figure Generate: Capire e Leggere i Visual

Questa pipeline di machine learning tabulare crea un repertorio diagnostico e grafico che compone l'output finale nella subfolder Report/ Fig ed Output/png. 
Ecco cosa produce ogni singola e magnifica esecuzione grafica del sistema ed il reale significato del Machine Learning dietro a ciò: 

- **Analysis_01_Target_ACF.png (Autocorrelation Function Plot del Consumo)**
*Cos'è:* L'Autocorrelazione (ACF) plot è il battito cardiaco grafico che verifica la teoria base prima di iniziare lo spropositato tabulare di estrazione delle features: L'auto-somiglianza fra step e step nel tempo! Traccia su X il tempo sfasato (step/Lag: 1, 2, ...48..) e sulla Y la Correlazione tra 0 (Non simile o causale/casuale) a 1 o -1 (Matematicamente speculare!). 
*Cosa Aspettarsi:* Picchi positivi forti vicino a t-1! Poi le curve calano progressivamente allontanandosi e, sorpresa visiva eccezionale: un'impennata gigante risorge come onda d'urto del terremoto verso esattamente la tacca "48 Lags", simboleggiando e dimostrando visivamente all'uomo Analista che l'indizio a 24 Ore esatte retrostanti di Orario Consuntivo (ieri ora identica rispetto all'oggi) ricalcola una Correlazione Assoluta Statistica pazzesca rispetto ad ore sbandate come la t-12 ad esempio. Un grafico dimostratore formidabile e sigillo d'utilità sulla costanza globale di impostare e di investire sui Regression Lag multipli anzideché singoli approcci lineari deboli. Muro Rosso Dashed d'interruzione stampato ed etichettato graficamente per non fuggire occhio.

- **Analysis_01_TimeSeries.png (Grafico Reale vs Baseline Prevista su Test e Test denormalizzato orario e datario)**  
*Descrizione e Lettura:* Grafico `plotRegressionResults`. Mette a sistema la Predizione Regression "ROSSA" o tratteggiato col Line Real Truth Black. Essendo su set di blocchi interrotti in date si prefigge di farti guardare i periodi che il programma (Essendo in Test di non visione globale) ha sfidato "In the wild". Le differenze della zona 9 su Triestino per il Regression Learner con Ensemble sono incredibilmente abili nel ricopiare la figura formale a forma di dromedario bi-gobba giornaliera. *Cosa Leggere*: Il problema emerge all'analista vedendo con facilità le "punte o acconciature supreme" di spiga che in rosso sono spesso limate più in basso delle Vette nere (Undeshooting the Peak). I modelli Regression Forest ed LSBoost tagliano spesso con la mannaia le medie fogliari ed ignorano per timidezza del rumore i super spikes violenti del picco assoluto, tenendo sicura la media e non andando ad aggredire.

- **Analysis_02_Scatter.png (Il Piano Cartesiano Predetto Contro Reale Consuntivo Punti)** 
*Cos'è:* Punti pallini trasparenti sparsi. Sull'asse X il Kw/h erogato dalla centralina V2G vera, sull'Asse Y il Kw/h scommesso ed indovinato dal Modello Regression.
*Come Leggere:* Linea Bisettrice Rossa Diagonale da Spigolo Basso a Spigolo Alto. Più le nuvole Blue (Pallini e stime in intersezione del grafico orario vs predetto) si ammucchiano come api attorno allo sciame lineare della diagonale Rossa (y = x, Perfetta Intersenza Identica Senza Scarti!) più R-Squared tende a salire ed il modello è robusto. La svasatura ovale orizzontale fa notare l'ammontare e lo sbordare d'errore sui punti più distanti, ad indicare l'ammassarsi su volti bassi o lo spargersi casuale ed incerto verso l'apice destro (consumo maggiore ed altamente sfuggente). Uno sguardo dice più di 10 coefficienti ed equazioni matematiche.

- **Analysis_03_Residuals.png (Subplots Multipli - Dashboard dell'Errore e della Statistica Residudiale)**
*Cosa contano queste 4 mini grafiche interne?* 
   1. The Subplot *Residuals Over Time.* Manda uno scatter cronologico con lo "Sbaglio e Zero" centrato. Indica con palese crudezza gli sbandi: se è sempre e spesso negativo sotto lo zero il modello Underfitta! Prevede roba più bassa di quella che accade. Se casuale o alterno simmetrico sui positivi stiamo godendo della natura casuale dello sbaglio, che per un calcolatore è sintomo di salute statistica su previsione causale.
   2. *Histograms Residuals (Istogramma o Campana Errore)*: Un muro sfumato verticale verde. Si vuole visualizzare il Gaussian Spread dei propri errori. La moda (Altezza Gigante) dovrebbe spingere allo ZERO asse X di devianza, srotolando in bassorilievi e scalinate sulle differenzianze ampie.
   3. *Auto-correlazione dello Sbaglio (Errore!)* Unica cosa da sapere: se le stampe non passano le linee arancio tratteggiate a terra piatte, siamo contenti! L'errore non porta e non si trascina schemi o colpe storiche. Lo sbaglio è accidentale e scorporato. Ma se c'è grossa correlazione sui lags dello sbaglio.. il modello si comporta systematicamente male e fallisce su pattern a non decodificati (Errore stupido sistematico). 
   4. Q-Q Plot a retta. Analizza la "Normalità Matematica e Normale Assoluta Gaussian-Like" rispetto alla linearità quantilica perfetta teorica. Qualità avanzata.

- **Analysis_04_HourlyError (Barrato di scarti temporali massivi in errore Orario - Diagnosi Orologio)**  
Il cacciatore di anomalie civiche. Mostra tramite stanghette in barre classiche di color Aranciato la quantità del Kw/H mancato (Sbagliato in termini assoluti `Errore Assoluto abs`) accorpati e condensati in media e spalmati verticalmente su tutte e 24 le sezioni dell'Orologio universale e plottati ed accesi contro i Deviazione Standard ErrorBars a picchetto "Nero" a raggio sulle punte!
*Da Aspettarsi*: Nelle ore fondissime (Ore Piccole Dalla Sub Notte all'Alba mattiniera ore 27 e 04 AM): Asticelle minuscole al punto da sparire dal grafico, che indicano e validano ed onorano come la persistente inerzia dei computer di casa e dei router a Roma e frigoriferi sia perfettamente predecibile dal Computer non producendo Errori al Kilowatt ed indovinando con una placidità pazzesca ed insindacabile! Contro un innalzarsi a scogli violenti sulle ore 9:000 e le 180:00 (Rientri Lavoro e Mattinate Fabbrichevoli ed uffici!) dove gli spaccati della casualità Umana deviano il consumo in maniera aggressiva battendo ed indebolendo fortemente i Boost di alberi o Regressione!

---

## 11. I Log Salvati: Struttura e Interpretazione

Tutti questi calcoli non scompaiono nel terminale transitorio della console o si smarriscono al riavviare del core, ma una rete rigida implementativa in file di Log e stringhe si materializza in salvataggio:

**Directory generata nel root e gerarchia:**
La super-cartella creata col Path unito alle variabili è:
`[Root]/Sessioni/Anno_Mese_DataOdorna/Report_regEnsembleAutoReg[OraEsecuzione]/`

Cosa c'è dentro lo scrigno creato ed inserito a completamento Task ed Epoca del Main?
-   Cartella Singola `fig/` = Presidio e deposito delle stampe digitali non bidimensionali "Statiche da Export .Png", ma formato .FIG crudo MATLAB e ricchissimo di sorgenti dati interagibili ed insediabili per futuri post-processing di design d'esame accademica. (I punti, i layers visivi ri-esplorabili a piacere per variazioni spessori etc).
-   File Png multipli sparsi `.png` con le classiche sigle testuali autoesplicative che compongono un referto medico radiologico: *Analysis_01, 02.. etc.* o per Regressione Zone multiple lo spillarsi in Loop di referti zona indipendenti denominati elegantemente via format Sprint: `"Zone_08_01_Scatter.png"` 
-   **La Perla Testuale - Files di Rapporto (.TXT o Diagnostics Diarizzati):**
`Analysis_Log.txt` o `Predictions_Log.txt`. Usando il comando astratto pre inglobato "DIARY ON/OFF" le variabili print terminale generano il fascicolo:
 *Cosa Contengono?* Lo spaccato sezionato ed esteso, l'analitica della stringa di codice base riprodotta con l'out-puts in chiaro:
- Numeri input Lag "Width", Modelli eseguiti.
- Metric Evaluation di R ed Errori Quadrati e Percentuali Mape sulle fasi (Train Test/Validazioni e Persistenza Fissi), per avere sottomano in qualsiasi momento documentabile di archivio l'esatte ed assolute matematiche d'errore formale dei Test (In formato numerico preciso non deducibile ad occhio dal Scatter Plot di fianco!), permettendo tabulazioni esterne con i concorrenti (Il file Excel degli analisti comparatori dei modelli mondiali V2g/Regression!). 
- Contiene inoltre *L'analisi OutLiers o dei Mostriciattoli D'errore dei peggiori dei Punti*. Questo script, presente nella parte finale del log di Regressione (Analysis), censisce non l'aggregato anonimo ma in particolare le *TOP 10 Candeline o Top 10 Orologi Pazzi test* e stampa in Testo nel txt ORA, DATARIO , Temperatura , VENTO per il peggior punto in assoluto precluso al modello (Il famoso Worst Case Scenario dove il Modello R2 ha spaccato colpa su tutte, predetto 85 a fronte di 220 erogato.. in genere a causa di perturbazioni, festività folli o cali eccezionali a black-out rete V2G!) dove gli scenari sono i "Cigni Neri". Avere il dump dei top cigni neri sul terminale a stringa isolata dona potere estremo di valutazione causa ed effetto pre analitico all'ingegnere senza sforacchiarsi nel visual o scavare array perduti! Spaventoso.

---

## 12. Analisi Critica, Limitazioni e Confronto con LSTM

Giunti all'estremo dell'analisi è tempo di bilancio critico. Quali sono i poteri asimmmetrici di queste strutture?

### Pregi e Vittorie Regression Learner sull'LSTM
1.  **Velocità ed Efficienza Energetica Assoluta.** Mentre allenare e ottimizzare su 100 iterazioni tramite Baesyana uno stack profondissimo Memory Cell ed Hidden State (RNN) impiega svariato tempaggio intensivo o necessità un Hardware in parallelo/GPU, LSC-Boost Regression / Random Tree Regression abbatte e chiude i loop su insiemi spropositati testuali in maniera incredibilmente violenta ed asciutta, consentendo agilità all'uomo per test continui ("Fast iterations di Sviluppo e Try/Error Tuning").
2.  **No Exploding e Vanishing Gradients e Over Complicazioni Ocultate**: A patto di usare alberi di decisione per l'Ensemble le feature sbrodolatissime "Lags -48" godono la facilità immensa ed inimmaginabile di superare ed inibire l'oscuro "Stato Segreto C" o pesi misteriosi che collassano nelle fasi backward propagation e back-passing dell'ottimizzazione del deep learning e minimizzazione su derivative e tendono su alberi con intersezioni binarie o su rette iperspaziali a minimi quadrati. Spesso non fallisce in instabilità per minuterie errate in Learning Rate (Che paralizzano i Recurrent Models e fanno fallire loss functions verso Infinite O Zero Loss).
3.  **Cross-V2G Zone Stability a Rischio Zero O quasi**: Come evidente le metriche ed il testing Cross-Generalizzazione dimostra una solida resilienza dell'Ensemblo a "Scavalcare le Contee e Città Geografiche Zone 9 ->10 e 11". L'adattamento ai Lag (Staccato su orari e giorni) garantisce fortissima coesione al Modello Regression Learner base al "Modello Baseline Semplice" Persistence della logica Y = Y_vecchio, sfondandolo con costanza anche se non mostrum.

### Limitazioni Forti Intrinsecate ed il Rovesciamento d'Assurdo (Contro LSTM)
C'è un motivo se Lstm ha rivoluzionato tutto.

1.  **Il "Curse Of Dimensionality" E lo Sbraccimento Spaziale (Larghezza di memoria vs Feature):**
Aumentando a dismisura in un futuro a risoluzione di orizzonte per giorni intercedenti non a 24 ore e quindi Mese e Mesi (Es test a 2 e 4 Settimane Retro per 30 min).. NumLags passerebbe da modici 48 a 720 (E Feature 9 a migliaia). I 432 array larghi in `createRegressionlags` finirebbero a colossali matrici immense e vacue per colonne, schiantando "Svm" a Rbf a memoria collassata. Un Modello Deep LSTM se ne esula. Si immettono i 9 input come una bocca del fiume costante. Che ne durino un mese, un anno o dieci minuti lo Stack della RNN macina un ciclo unitario senza srotolare nulla con la tecnica formale dello squence memory cell ad una cella loop ed iterabile a peso ridotto (Celle o unità Neuroni interni che si aggiornano singolarmente e limitatatamente!). L'LSTM non esplode geometricamente nella larghezza. 
2.  L'assoluto non allineamento Logico del tempo. "Vento(T-48)" per il Boosted Trees o Random Forest Modello è al pari di uno slegato attributo alieno come il "Nome Città!" come una Categorical Random Feature e l'informazione in colonna A vs Colonna C è non direzionale per sua matrice ed essenza base (Nessuna freccia relazionale unire la colonna del vento ieri e ventò oggi per lui. E' un puzzle ammucchiato piatto). Spetta ai taglierini (Split ed Iterazioni albero o Rbf Kernel Spaziale a retta) fare innumerevoli connessioni astratte per non curanza o di non sapere i legami della sequenzialità innata per cui è stata spezzata!

Questo documento analizza dettagliatamente come la sfida Regression affrontata con maestria d'Ensemble affronti magnificamente questa colossale e spinosa perdita del legame fisico logico originario col tempo, avvalendosi della robustezza srotolativa su Z-Scoring massivo ed intelligente ri-sottomissioni cross over e predizione testuale ad intelligenza selettiva, divenendo un degno ed ottimo rivale (o Benchmark Zero a Parità) da sfidare in maniera solida in accademia col V2G Forecasting al reame del Neural Stack Network più moderni e blasonati LSTM sequence to sequence architectures che sono padroni e signori incontastati recenti del dominio temporale profondo non lineare multivariabile moderno.

---

*Analisi Antigravity AI, Marzo 2026. L'indagine e documentazione dettagliatissima da Codice Regression.m fino alle predizioni Generalizzate e Analisi Log File V2G Test Zone.*
