# Analisi Tecnica Completa del Progetto: V2G con Reti Neurali Recorrenti (LSTM Seq2Seq)

**Autore:** Analisi generata da Antigravity AI  
**Data:** Marzo 2026  
**Progetto:** V2G-with-Regression-Learner-and-LSTM-NN  

---

## Indice

1. [Contesto Applicativo: Vehicle-to-Grid (V2G) (Partendo da Zero)](#1-contesto-applicativo-vehicle-to-grid-v2g-partendo-da-zero)
2. [Il Problema: Previsione del Consumo Energetico](#2-il-problema-previsione-del-consumo-energetico)
3. [Paradigma di Risoluzione: Reti LSTM e Modalità Seq2Seq](#3-paradigma-di-risoluzione-reti-lstm-e-modalità-seq2seq)
4. [Architettura del Codice e Flusso di Esecuzione](#4-architettura-del-codice-e-flusso-di-esecuzione)
5. [Struttura e Costruzione dei Dati: La Creazione delle Sequenze](#5-struttura-e-costruzione-dei-dati-la-creazione-delle-sequenze)
6. [Analisi Dettagliata degli Script Core](#6-analisi-dettagliata-degli-script-core)
7. [L'Intelligenza Artificiale al Lavoro: Bayesian Optimization](#7-lintelligenza-artificiale-al-lavoro-bayesian-optimization)
8. [Previsione e Generalizzazione Cross-Zona](#8-previsione-e-generalizzazione-cross-zona)
9. [Le Figure Generate: Capire e Leggere i Visual](#9-le-figure-generate-capire-e-leggere-i-visual)
10. [I Log Salvati: Struttura e Interpretazione](#10-i-log-salvati-struttura-e-interpretazione)
11. [Analisi Critica, Limitazioni e Conclusioni](#11-analisi-critica-limitazioni-e-conclusioni)

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

Pertanto, un modello di previsione accurato del consumo elettrico (Load Forecasting) è l'enzima fondamentale per le logiche del V2G. Questo documento esplora l'approccio più avanzato a questo problema: l'uso di Reti Neurali Ricorrenti di tipo LSTM.

### L'Ambito di Progetto
Il progetto si concentra sul prevendere in maniera quantitativa (kWh) il fabbisogno energetico aggregato di porzioni urbane precise della città di **Roma, anno 2023**, utilizzando il Deep Learning. Le porzioni urbane sono state suddivise in "Zone" codificate numericamente (Zona 8: Anagnina, Zona 9: Trieste, Zona 10 e 11: Tor di Quinto e aree limitrofe). I dati sono ad elevata frequenza: risoluzione oraria di **30 minuti** (half-hourly).

---

## 2. Il Problema: Previsione del Consumo Energetico

### Analisi del Problema
Prevedere il consumo è un problema matematico inquadrabile come **regressione su serie temporali** (Time-Series Regression). Stiamo cercando di prevedere una variabile continua target $y$ (consumo elettrico in kWh al tempo $t$, ovvero `AAC_energy`) basandoci su informazioni storiche e descrittive.

Il consumo di energia non è un fenomeno casuale (white noise), ma mostra forti componenti sistemiche:
- **Stagionalità Giornaliera:** Ciclo continuo notte-giorno. Picchi mattutini al risveglio, picchi serali al rientro lavorativo, valli notturne profonde.
- **Stagionalità Settimanale:** I giorni feriali (lunedì-venerdì) sono molto diversi dai weekend (sabato-domenica).
- **Auto-correlazione forte (Inerzia Termica ed Elettrica):** Il consumo al tempo $t$ è fortemente correlato al consumo al tempo $t-1$ o $t-48$ (lo stesso momento del giorno precedente).
- **Fattori Esogeni (Meteo):** Se l'aria condizionata è accesa (temperatura alta d'estate) o se accendono pompe di calore (bassa d'inverno), il consumo schizza. Anche fenomeni come la precipitazione inducono le persone a stare a casa, aumentando il consumo domestico.

### Dataset a Disposizione ed Estratti
Per catturare le suddette dinamiche si possiedono diversi flussi di informazione (Predittori/Features) fusi assieme:
1. **Dati Autoregressivi / Endogeni:** `AAC_energy`, il consumo. Si utilizza la storia del consumo stesso per capire il trend attuale.
2. **Dati Meteorologici:** `temp` (Temperatura in C°), `windspeed` (Vento in km/h), `precipprob` (Probabilità pioggia in %).
3. **Dati di Calendario:** `holiday_indicator` (Boleano 0/1 per giorni festivi).
4. **Dati Tempo-Ciclico:** `hour_sin`, `hour_cos`, `day_sin`, `day_cos`. Il tempo è circolare. Il mese e le ore del giorno non sono rette ma curve chiuse. Attraverso seno e coseno l'orario viene "avvolto" su un cerchio, per permettere all'algoritmo di decodificarne la continuità.

**La Sfida del Dataset Frammentato:**
I dati **non coprono l'intero anno 2023 in modo continuo**. Sono presenti "cluster" temporali (finestre di dati disponibili), con gap di settimane tra uno e l'altro. Questo rende impossibile l'addestramento su un intero ciclo stagionale ininterrotto, costringendo la rete ad imparare pattern locali super-ottimizzati anziché macro-trend annuali.

---

## 3. Paradigma di Risoluzione: Reti LSTM e Modalità Seq2Seq

A differenza dei modelli tabulari (come i Random Forest o gli SVM usati in `RegressionLearner.m`) che non hanno memoria intrinseca e vedono i dati "piatti", le reti **LSTM (Long Short-Term Memory)** sono progettate specificamente per flussi temporali.

### Anatomia di una LSTM e la "Memoria"
L'LSTM è un'architettura di Rete Neurale Ricorrente (RNN). Invece di far passare il dato da In a Out e fine della storia, le LSTM hanno dei "loop" interni. Quando processano il dato al tempo $t$, non guardano solo l'input corrente $x_t$, ma tengono conto di una variabile nascosta $h_{t-1}$ (Hidden State) e $C_{t-1}$ (Cell State) che viene passata dal calcolo dell'istante precedente.

Questo stato funge da "nastro trasportatore della memoria". L'LSTM decide cosa ricordare e cosa dimenticare tramite complessi ingressi matematici chiamati "Gates" (Porte):
*   **Forget Gate**: "Devo dimenticare i dati della mattina ora che è sera?"
*   **Input Gate**: "L'informazione sul temporale appena iniziato è vitale, memorizzala nello stato cellulare!"
*   **Output Gate**: "Dato che è festivo e la memoria dice che piove, emetti un segnale di alto consumo casalingo".

### Sequence-to-Sequence (Seq2Seq): Il Cuore del Progetto
Questo è il concetto più rivoluzionario implementato nello script `LSTM.m`. Spesso, le reti temporali vengono addestrate in modalità "Sequence-to-One": diamo 48 ore di dati in pasto allo stream, la rete macina, calcola, respira, e alla ultimissima ora emette "1" singolo numero: la predizione futura. 
Questo approccio su un dataset con soli 56 giorni (pochi campioni) porterebbe a un addestramento misero.

Il progetto vira audacemente sul pattern **Sequence-to-Sequence (Seq2Seq)**. 
Con questo approccio, la rete non aspetta la fine per imparare. Ad *ogni singolo passo temporale*, mentre la sequenza scorre dentro la LSTM, la rete viene forzata a sbilanciarsi e prevedere il passo immediatamente successivo:
*   Prende T=1 -> Prevede T=2. Sbaglia, viene bacchettata ("Backpropagation").
*   Prende T=2 (+ la memoria di T1) -> Prevede T=3.
*   Eccetera...

Questo passaggio si evince nel codice con l'istruzione `lstmLayer(..., OutputMode="sequence")`. Questo setup moltiplica esponenzialmente il segnale di errore e i dati di addestramento: una sequenza continua di 10 giorni non fornisce "1" punto di addestramento per la fine, fornisce 480 punti di addestramento distribuiti temporalmente!

---

## 4. Architettura del Codice e Flusso di Esecuzione

L'architettura orchestrata in `LSTM.m` è meticolosamente progettata per evitare le insidie statistiche (Data Leakage) e massimizzare l'oggettività scientifica dei test. Ecco il percorso logico:

1.  **Caricamento Dati e Normalizzazione (`loadZoneData`):** Fusione dei file meteo, calendario e consumo.
2.  **Toggle Constants e Variabili di Sistema:** Decisione pre-addestramento (Uso Bayesiana? Feature auto regressive attivate?).
3.  **Il Selezionatore Paziente (`selectRepresentativeDays`):** Questo è il cervello del testing leale. Siccome i dati sono spezzati (Cluster 1 a Febbraio, Cluster 2 a Maggio, ecc), prelevare giorni a caso per testare il modello rischierebbe di fare testare il sistema solo su calde estati saltando la dura prova dell'inverno. Questo script "sceglie chirurgicamente" 2 giorni rappresentativi (Una domenica, un giorno normale mediano e climaticamente stabile) spalmati sui vari gap per garantire un Validation e Test set equo che copra tutte le situazioni stagionali. Lo estrae brutalmente ed inesorabilmente dal dataset, accecando la rete neurale su queste date in modo totale.
4.  **Codifica Orologio e Splitting:** Costruzione delle sinusoidi orarie e rimozione ufficiale delle righe. (`splitTrainValTest`).
5.  **Normalizzazione Z-Score a "Tenuta Stagna" (`normalizeZScore`):** Calcola la Curva di Gauss (Media e Scalatura a Devianza Standard) **ESCLUSIVAMENTE** sui dati che la rete vedrà in Training. Memorizza quei coefficienti magici in `normParams`. Poi, in modo asettico, applica quella media a Validation e Test. Se calcolassimo Media Generale del consumo includendo il test, il modello Deep Learning, nella sua diabolica bravura, sosterrebbe il test indovinando basandosi sul baricentro compromesso dall'inizio. Protezione Anti-Leakage Perfetta.
6.  **Il Tagliatore di Pellicole (`createLstmSequences`):** Costruisce i "Film" temporali (vedi sezione dedicata a seguire).
7.  **Scontro con la Macchina o Default (Optimization / Architecture):** La strada si biforca tra Auto-tuning o architettura manuale fissa.
8.  **Training Core (Il Forno):** Usa `trainnet` con l'algoritmo adattivo `Adam Optimizer`. Qui la Backpropagation fa miliardi di calcoli riducendo il parametro "MSE" (Errore Medio Quadratico). 
9.  **Predictions e Disimballaggio (`minibatchpredict` e denormalizzazione):** Usa la rete per indovinare; toglie il padding (gli zeri usati per allineare vettori non gemelli) e applica la moltiplicazione per la Devianza per far tornare i risultati nella vita reale al valore "Kilowatt/Ora".
10. **Metrologia e Diagnostica Visiva (`computeMetrics` e `plotResults`):** Genera stampe finali. Salva i modelli compressi e crittografati in chiavetta per esecuzioni successive (`saveModels`).
11. **Rapporti Testuali automatizzati (`diary on/off`):** Manda in spool ogni testo della shell nella cartella `Sessioni`, utile per redigere pubblicazioni senza ricopiare dal terminale.

---

## 5. Struttura e Costruzione dei Dati: La Creazione delle Sequenze

La manipolazione geometrica del dato dentro le LSTM è radicalmente diversa dal Machine Learning tabulare. Qui interviene l'importantissimo script `createLstmSequences.m`.

Mentre in `RegressionLearner` i dati si "sciancano" srotolando le vecchie 48 ore in orizzontale su 432 colonne statiche, per la rete LSTM i dati restano col formato nativo "Sottile e Lungo" (es. 9 colonne per feature, X mila righe per lo step temporale).  
Tuttavia c'è un grosso problema applicativo: i gap temporali e i buchi neri.

**L'Algoritmo `createLstmSequences`:**
Questa funzione riceve la tabella cronologica e un ordine: "Cerca continuità!".
Siccome abbiamo dati sfaccettati ed asimmetrici per colpa del dataset originario con buchi, e abbiamo perfino creato "voragini" intenzionali portando via a forza giorni di test.. la rete non può leggere `Lunedì 10 Mese`, poi alla riga due leggere `Sabato 22 Mese Seguente` ed ipotizzare che sia passata mezz'ora! La rete LSTM, fidandosi del loop di memoria, andrebbe in allucinazione collegando climi e stagioni sfalsati infettando la *Cell State* di memoria irrimediabilmente.

**Cosa fa lo script:**
1.  Spazzola la riga temporale `time_vector`. Cerca `diff()` (i delta T tracciati).
2.  Se trova un gap superiore a 35 minuti, aziona la ghigliottina virtuale (`breakPoints`). Il dato si trancia!
3.  Crea un "Blocco Continuo" e lo impacchetta col formato dei Cell Arrays in Matlab `XTemp{validCount}` . 
4.  Siccome iteriamo la strada suprema *Seq2Seq*, per ogni blocco il ritaglio è sfalsato:
    *   La matrice `X` prende i valori dalla stringa $t=1$ a $t=Fine-1$.
    *   La matrice target `Y` si allinea alla perfezione prendendo il dato da $t=2$ a $t=Fine$.
    Così la Rete ad ogni colpo riceve "Dato che sai il Passato e il presente ($X$), impara a darmi il futuro di 1 Step avanti ($Y$)".

Se il dataset genera 7 Blocchi continui su Training, l'Input Cell per il train (`xTrain`) consisterà di 7 Vettori Multidimensionali asimmetrici (e.g. $[200 \times 9]$, $[800 \times 9]$, $[400 \times 9]$). Questa architettura permette alla LSTM di divorare stringhe temporali pure e non artefatte su tabulazione artificiale.

---

## 6. Analisi Dettagliata degli Script Core

Andiamo sotto il cofano dell'architettura in `LSTM.m`.  

### Architettura della Rete (Layers)
```matlab
layers = [sequenceInputLayer(numFeatures, Normalization="none")];
for i = 1:numLayers
    layers = [layers
        lstmLayer(numHiddenUnits, OutputMode="sequence")];
end
layers = [layers
    fullyConnectedLayer(100)
    reluLayer()
    dropoutLayer(drop)
    fullyConnectedLayer(numResponses)];
```  

**Perché questa forma per l'architettura neurale?**
Non è casuale ed è lo state-of-the-art per la "Time Series Regression" (Raccomandato ufficilamente da MathWorks su Forecasting Seq2Seq). Questa forma prende i flussi sequenziali (es. 9 feature in Input) e li processa con un'apertura massiccia a cono (il Layer LSTM che solitamente con l'ottimizzazione è balzata a $\approx 119$ unità "neuroni nascosti" o *Hidden Units*). `OutputMode="sequence"` lo diciamo ancora, ordina la trasmissione costante e sputa un array di 119 proiezioni continue per ogni frame di tempo.
A seguire le rappresentazioni passano dentro una sorta di imbuto (Bottleneck design): `fullyConnectedLayer(100)`. Cento unità dense fisse. Serve ad accorpare i mega dati latenti sparsi dalle celle dell'LSTM in un set concentrato e meno dispersivo.

**L'attivazione e La Cura all'Overfitting:**
Il `reluLayer()` (Rectifier Linear Unit: toglie tutto ciò che è negativo e lascia i positivi intatti). Toglie la linearità morta ed insegna alla rete l'ingegno per interpretare "picchi improvvisi matematici anomali" impossibili su scala lineare.
Infine, `dropoutLayer`. Quest'arma magnifica "spegne" randomicamente (col drop Rate del 33%) alcuni neuroni dell'imbuto durante ogni giro di apprendimento e "Backpropagation". A cosa serve? Ad evitare che la rete diventi pigra ed in affanno impiegando solo un paio di collegamenti eccellenti come fossero la via facile "Shortcut Autoregressivo" a studiare i consumi a memoria e copiare i compiti. Senza che lo sappia, un terzo del cervello le viene azzerato costringendola incessantemente a far circolare il suo ragionamento sulle variazioni climatiche anche se il "copia incolla temporale" era molto affidabile! Infine si spunta su `fullyConnectedLayer(1)` che decanta il risultato su unica colonna predittiva "Aac Energy Target Kw".

### Training Options (Opzioni di Addestramento)
Il motore dell'apprendimento su `trainingOptions`. 
- **L'Ottimizzatore `adam`:** Il re dell'addestramento, ricalcola i pesi neurali adattando la frenata man a mano che il traguardo è vicino senza schiantarsi sulla Loss per troppo ardore a differenza del gradient normale. E' rapidissimo. 
- **Batch Size Intelligente:** `MiniBatchSize` solitamente a $\approx 30$. Invece di aggiornare gli errori dopo aver visionato l'intero film dell'anno solare, la rete calcola e modifica la memoria in segmentini continui per un veloce "fine-tuning". 
- **Imbottitura Lato Sinistro (`SequencePaddingDirection="left"`):** Cruciale. Le sequenze in Input non sono della stessa lunghezza nel matrix in calcolo di batch! Una ha 120 giorni.. l'altra 20 giorni perché mozzata da Cluster rotti. Come imbutare due mattoni non uguali in una fornace rettangolare della GPU Nvidia che vuole matrici quadrate? Il padding "Zero" (aggiungere zeri). Posizionarlo a SINISTRA ("left") anziché al fondo (destra) assicura che il dato più critico e vicino e vibrante e rilevabile (Ieri, Oggi) sia accostato a pelo sulla lente di "t=0" d'inferenza ed output finale non sprecando energia neurale in decodifiche su spazi bianchi in mezzo! 
- **Apprendimento Pieghevole (`piecewise`):** Inizia impacciato al `0.01` di LearningRate (Corre veloce ma fa sviste enormi) per fare la prima esplorazione colossiera nel campo. Ogni $50$ epoche frena ed abbatte "Drop Factor a 0.2", sminuendo d'un quinto l'incisività e scendendo a passo d'uomo al rallentatore per far depositare il risultato con la mano vellutata al momento d'arrivare al "Centro perfetto dell'algoritmo" per scansare le imperfezioni decimali micro-KW senza disfasare l'ottimo globale locale! La genialità della discesa del gradiente.

---

## 7. L'Intelligenza Artificiale al Lavoro: Bayesian Optimization

Questo progetto contiene, se la toggle `useBayesianOptimization` è attiva, un gioiello d'ingegneria che pone una I.A. per allenare ed arginare i dilemmi di base dell'architetto della Rete (Data Scientist).
L'analista si chiede: *"Qual'è il bilanciamento idilliaco della Rete? Mettere 2 o 3 layer profonidissimi a cascata LSTM aumenterà in complessità intrapolando l'Universo temporale astronomico, oppure andrà a creare grovigli mentali ridondanti overfittando e fracassandosi sull'addestramento come mattoni pesantissimi sfatti di overfitting su soli 56 giorni?"* Ed il Dropout a quanto lo setto?

Esiste la via brutale del "Lascio girare per mesi il Grid Search testando 10 mila combinazioni da 0.1, 0.2...". Esiste la via di Bayes (Implementata). 
Nel file si attiva `bayesopt()`. Non prova tutto a casaccio. Crea una Mappa surrogata Gaussiana probabilistica sulle combinazioni possibili dello spazio intero a 5 dimensioni (Layer, Batch, Neuroni, LearnRate e Dropout).  
Il codice manda la rete a fare mini-test leggeri e rapidi (`lstmObjectiveFunction` esegue a sole 50 mini epoche compresse). Ottiene un errore (RMSE di Validation). Invece di farne mille, l'ottimizzatore Bayesiano guarda le perdite passate.. suppone.. sbilancia in avanti il suo raggio d'esplorazione (Acquisition Function `expected-improvement-plus` in cerca dell'Aurea Media bilanciando Exploration/Exploitation). Cerca di affondarsi e calibrare le opzioni solo verso terreni che odorano di bassi rate di sbagli ed ignora in blocco i calcoli massicciamente folli. Dopo solo 100 TENTATIVI produce la topografia ottimale eccelsa.

**I Risultati Sorprendenti di Bayes (Nel nostro setup consolidato standard):**
- Modello con **1 SINGOLO LAYER** è sempre risultato il campione indiscusso che distrugge configurazioni Deep Network a 3 o 4 Strati ricorsivi! Sorpresa per gli umanoidi che tendono al "Deep & Complex". Dimostra con audace ferocia come "Small Data Needs Shallow Model". Troppi strati LSTM su dati così stretti non affinavano l'orologio umano bensì memorizzavano ciecamente la sequenzialità a vuoto.
- Neuroni Ideali `~119`.
- Lo scheduler di Dropout (0.33) vince con una proporzionalità di stallo d'overfitting fenomenale. 
Tutti i valori Default impostati a freddo provengono da questo giudice inossidabile applicato sul campo romano di Trieste e Anagnina.

---

## 8. Previsione e Generalizzazione Cross-Zona

### Il Giudizio Finale: `LSTM_Predictions.m`
La ricerca ingegneristica non finisce su un plot bello sul Validation e Test "Estratto della solita zona". 
Il file indipendente `LSTM_Predictions.m` agisce. Questo script è un distruttore di false speranze ed allucinazioni neurali. E' il vero Benchmark industriale e professionale per testare un Modello IA Load Forecasting.

Questo strumento carica l'ultimo Super Modello allenato da un momento casuale (es. Zona 9 e le sue devianze asimmetriche calcolate via Z-Score di Trieste e via d'uscita archiviate nell'enciclopedico `Models_....mat`). 
Senza batter ciglio lo script inizia un ciclico perlustramento forzato sui territori limitrofi romani (`targetZones = [8, 9, 10, 11]`).

**L'Architettura del Test Cross-Zone:**
1.  Va in Zona 8 Torvergata (Dati Mai visti dalla matrice di rete LSTM nei pesi neuroni). 
2.  Normalizza gli scalari in base ai pesi locali e decodifica e trincia il tutto a matrice `LSTM Sequence X/Y Target`.
3.  Esegue Batch Inferenziale al buio: (`yPredNormCell = minibatchpredict`).
4.  Cosa cerca di capire? Valuta se "A prescindere dall'edificio... Le logiche meteo impattate all'umano che usa frigo d'inverno a domenica, associate a orologi pendolari della mattina.." sono state assorbite come intelligenza generale!

Questa operazione spinge la rete neurale a scavalcare la Persistenza. Il report del terminale automatico ci dirà brutalmente in tabella se sconfigge ($NO/YES$) Baseline. Il calcolatore in fine eroga persino giudizi espliciti ad if/else di Quality Control sulle fasce fallaci ed anomale (Peak error).

---

## 9. Le Figure Generate: Capire e Leggere i Visual

Le LSTM generano dashboard fondamentali per l'umano tramite le esecuzioni visive ed automatizzate salvate poi nel folder `Sessioni/Report../fig(e PNG)`:

- **Analysis_01_Target_ACF.png (Autocorrelation Function Plot)**  
*Cosa Vedo:* Barre azzurre pendenti su Assi Lag in "Scala Step Di Mezz'Ora". Le fasce orizzontali Dashed Marroni sono limiti 95% Confidence Bounds Statistic. 
*Come Interpretarlo:* Sulla serie nativa, si vede una campana ciclica che si alza, cala a zero verso i lag $\sim24$ (Ora opposta del giorno, Notte contro Giorno a coefficiente nullo e negativo su incrostanza opposta) e rinasce portentosa sparando i picchi quasi da $0.80$ verso e sopraggiungendo contro l'asterisco `Lag 48` posizionato in Rosso. A dimostrare che a un T-Minus Day il consumatore e l'impronta elettrica termodinamica V2G ritorna a presentare un profilo matematicamente affine a correlazione altissima, fondando le decisioni degli step passate e precludendo e conclamando lo sviluppo e fondazione di logiche retro attive che l'LSTM ingoia da passati a lungo terminanti.

- **Sequence_03_Test.png (Time Series Reale Contro Predetto dell'Inferenziale LSTM)**  
*Descrizione:* L'implicita `plotResults.m` fonde ed alloca denormalizzate le super uscite sul blocco predittivo. Sulla cartella i quadratini neri uniti al tratto nero raffigurano cosa è stato "Erorgato davvero alle Cabine e Colonnine ed Abitazioni (Target veritieri)". La Linea color Rosso tracciata come un'ombra sopra raffigura la profezia della Rete sulle sequenze incrociate sui giorni del set di test.
*Cosa dedurne ad occhio:* Le sequenze e gap sono unificate con linea orizzontale Dashed Grigia in demarcazione date! (Quello è il ritaglio della rete saltante gap!). Si evince se la rete non si slancia troppo alle vette vertigiose mattutine fallendo l'altezza od il peso del "Spike Assurdo di Accensione Generale del Condominio e Riscaldamenti Commerciali" che alle $11.30AM$ sfuggono ad algoritmi lisci e conservativisti. E' l'indicatore principe di solidità all'interpolazione del segnale e non lineare e complesso e caotico della metropoli!

- **Analisi_02_Residuals.png (Pannelli Errore Residuano)**  
1. *Scatter su Test in Tempo:* Selezionati i test, disegna lo "Scarto di kW" in asse al giorno e ora. Si evince se la rete sbaglia grave su Gennaio o Maggio. Esempio, d'inverno sbaglia grossolano e d'estate l'aria condizionata produce errori piccoli con punte occasionali positive ad ondata. Se sbilancia l'asse $X=0$ in basso in continuo: la rete fa "Undershooting", ha paura di fare stime grosse e resta a livelli di guardia di stima bassa difensivistica e conservativa (Da ovviare!). 
2. *L'Acf dello Sbaglio:* Si spera con tutta l'onestà accademica possibile che questi stecchetti nel grafico n3 non sforino i limiti rossi tratteggiati! E' il check up medico finale e crudele. Se lo scostamento residuale possiede Autocorrelazioni statisticamente violente fuori banda, l'algoritmo non ha fallito per colpa "Naturale Insondabile di Variabilità Random Umana White Noise..". L'algoritmo ha "lasciato qualcosa sul tavolo". Una variabile importante e prediscibile sistemica non è stata fusa nel Deeplearning che sta faticando ad intenderla ma c'è uno sprono "Pattern Non Scoperto!" da risolvere nel modello e rivedere. (Punto su cui molte analisi ingegneristiche e ML V2G si arrovellano ore intere per risolvere o arginare senza bias).

- **Analisi_04_Hourly_Error.png**
Grafico in istogramma e bar chart di precisione ad una sbarra e retta errore Errorbar che seziona e sega ad accette e scarti statistici le 24 Ore Orologiche in maniera a-cronologica. Quale orario il Modello fallisce aspramente e va fuorigiri rispetto alle fasce e stime che per esso suonano perfette da indovinare ed assolute e facili ad interpolare? Spesso nelle RNN seq to seq come questo si evince (e si legge in alto a stampa diagnostico) picchi tra 9-11AM (Impegni Misti Casalingo + Negozi a rampa vertiginosa accensioni). Un bel grafico da incollare e presentare da Data Scientists in Report finali senza se e matti, cristallino da intendere per i direttori d'analisi dei Load Forecasters della Rete Italia/Mondo!

---

## 10. I Log Salvati: Struttura e Interpretazione

L'infrastruttura di reporting non lascia scampo alla volatilità di memoria dei terminal.

La creazione Directory si attesta su: `Sessioni / Data_Esatta_Esecuzione / Report_lstmAutoReg[Timestamp]`.
Il Log testuale `Analysis_Log.txt` emette a spool la cronistoria.
Tra le centinaia di righe incise spuntano come oasi le fasi salienti. Innanzitutto l'echo del Dataset: Dimensioni matrici seq2seq, Architettura del Deep Learning LSTM che decanta ed informa il reportista se le "Hidden unit 119 - ReLU ecc" ha agito.
In modo critico, al fondo appare il listato tabellare per l'esame: R², RMSE e MAE. Tutto rigorosamente incapsulato e printato per "Test Normals (Valori Normali Irrazionali senza senso visivo e pre-scalaggio devianza Z)" ed "TEST METRICS (Denormalizzati In KiloWatt Ora)" dove il tecnico energetico deve battere l'occhio ed annotare i suoi Kw di deviazione!
**L'Estrema Caccia Anomalie:** L'Outlier Tracker Print in Log stampa una decina di record finali mozzafiato. Identifica la "Top Dieci" di colpi fuori margine (I mega Errori massimali incappati nel set test). Li printa asserendone per giunta i pre requisiti scatenanti come Temperatura in quel momento (18 °? Pioveva Prob = 100%?). L'aiuto insostituito per il debug del fenomeno non appreso e della catastrofe meteorologica o d'orario imprevisto fuori soglia del Machine Learning senza ricontrollare su codice base matrice YTest! Meravigoloso!  Sulla Cross_Prediction si salverà pariteticamente "Prediction_Log" sviscerando i giudizi testuali per "Quality Of Output Excellent\Good\Acceptable o Delta Persistence". Tracciatura su pietra di ogni Run per tesi, presentazioni finali di Paper e ricerca senza rimorsi.

---

## 11. Analisi Critica, Limitazioni e Conclusioni

La sfida predizione carichi V2g Load Forecasting per una zona a Micro Scaglione Roma risulta tra le architetture Data Science tra e più ardite: non stiamo prevedendo "consumi del Nord Italia 25 GigaWatt e Costanti" che godono macro scale perfette di massa che piallano via lo sbalzo di un accensione forno da singola unità o quartiere. Stiamo gestendo micro zone rumorosissime (La persona X che accende asciugacapelli si legge nei micro grafici del cablaggio Locale!) ed incostanti! 
Ad aggravare la Rete c'è la scarsezza temporale: 57 Dati sfaldati bloccano il deep Learning dall'andare a spasso fluido sulle derivate annuali massicce di "Esempi Stagionali da studiare anni su e giu e pesate ed iterazioni".  

L'LSTM con modalità Seq2Seq architettato è difatti magnifico. La forma in cui avvolge e spezzetta a blocchi di trinciatura ed emette al Backpropagation in previsione ad Orizzonte su Step Singoli su sequenze sfalsate crea artificialmente tonnellato pesanti di campioni forzati utili ad addestrare LSTM profili in tempo brevissimo salvaguardando Overfitting a suon di Early-Stopping (Patience=20!)

**Conclusione Reale sulla Proiezione Contro l'Arte Pura Regression:** 
La Rete neurale ricorsiva ingurgita il tempo naturalmente e vi sguazza in profondità dimensionali lineari ad impatto tridimenzionale, catturando i trend logici relazionistici come la Regression base "schiacciata e Flattened" a blocchi tabellari ignorerebbe palesemente e che sfiancano macchine e memoria RAM per colpa di colonna "400X width" feature matrices!  
Purtuttavia non è la panacea d'ogni male. Con dati scarsi, un buon Ensemble di Alberi potenziato ed aggirando il "Data Copy Bug dell'auto-ragressone t-1" lo sorpassa su certi quadranti zonali, sfidando di merito e di muscoli ed esecutabilità il regno di Keras e Reti profonde e lente che impiegano venti e passa minuti ad ottimizzarsi senza estrapolare magie occulte laddove non vi siano macro correlazioni di settimane piene e dati spropositati che i colossi neurali anelano.
Siamo di fronte a un solido ed inossidabile e formidabile banco test di accademia statistica incapsulata a codice purissimo al servizio della ricarica intelligente degli Autoveicoli e Città! (Fine). 

---

*Ultima revisione: Marzo 2026. Antigravity AI Documentazione*
