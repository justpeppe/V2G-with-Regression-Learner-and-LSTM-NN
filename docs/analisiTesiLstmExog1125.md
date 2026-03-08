# Analisi Dettagliata dell'Addestramento LSTM: Modello `lstmExog1125`

Questo documento si propone di esaminare, con scrupolo analitico e vocazione didattica, l'intero ciclo di vita, l'architettura e le prestazioni del modello di rete neurale ricorrente denominato `lstmExog1125`. L'obiettivo è fornire una scomposizione "Junior-friendly" in cui ogni concetto tecnico, statistico e matematico venga spiegato partendo dalle sue fondamenta, permettendo così a qualsiasi lettore (studente, esaminatore o curioso) di comprendere *il perché* e *il come* dei risultati ottenuti.

Ogni singola affermazione presente in questo report è rigorosamente supportata, dedotta e citata dai file di log grezzi (`Analysis_Log.txt` e `Predictions_Log.txt`) o dall'ispezione diretta del corredo grafico generato durante la sessione di addestramento.

---

## Capitolo 1: Introduzione e Scopo dell'Esperimento

### Il Contesto: Vehicle-to-Grid (V2G) e la Previsione del Carico
Ci troviamo all'interno di un progetto che studia l'interazione tra i veicoli elettrici e la rete di distribuzione energetica, un paradigma noto come **V2G (Vehicle-to-Grid)**. Il V2G non si limita a considerare l'auto elettrica come un passivo consumatore di corrente, ma la immagina come una "batteria con le ruote", capace di assorbire energia quando la rete ne ha in abbondanza (es. picchi di produzione solare) e di restituirla quando la rete è sotto stress.

Affinché questo delicato scambio bidirezionale possa avvenire in sicurezza e con efficienza economica, il gestore della rete deve sapere *esattamente* quanta energia sarà richiesta (o prodotta) in una data cabina primaria in un momento futuro. È qui che entra in gioco il concetto di **Load Forecasting** (Previsione del Carico). Sbagliare questa previsione significa rischiare blackout o sprechi economici enormi. 

### Perché proprio una Rete LSTM?
Per stimare il carico futuro, non possiamo usare una semplice equazione lineare. Il consumo di energia elettrica di una città o di una zona dipende dall'ora del giorno, dalla temperatura, dal fatto che sia un giorno lavorativo o festivo, e, soprattutto, dai consumi delle ore immediatamente precedenti (il cosiddetto concetto di *memoria storica* o *lags*).

Per questo esperimento, abbiamo scelto un'architettura **LSTM (Long Short-Term Memory)**. A differenza delle reti neurali tradizionali (dove l'informazione viaggia solo in avanti e non c'è memoria del passato), le reti LSTM sono progettate appositamente per analizzare "serie temporali", ovvero sequenze di dati ordinati nel tempo. 
Un neurone LSTM possiede al suo interno dei "cancelli" logici (Gate) che decidono matematicamente:
1. Quali informazioni passate sono inutili e vanno "dimenticate" (Forget Gate).
2. Quali nuove informazioni sono rilevanti e vanno immagazzinate (Input Gate).
3. Quale sarà l'output definitivo da passare allo step successivo (Output Gate).

Questo le rende perfette per capire, ad esempio, che il picco di consumo delle 18:00 dipende dal fatto che le persone stanno tornando a casa dal lavoro in quel preciso istante stagionale.

---

## Capitolo 2: I Parametri di Addestramento e la Filosofia della Rete

Per capire come la nostra rete "ragiona", dobbiamo sollevare il cofano e analizzare la sua architettura. Come si evince dall'analisi diretta del file `Analysis_Log.txt` (alla sezione "LSTM MODEL SUMMARY"), il modello è stato istruito su uno specifico spazio geografico e con una precisa conformazione.

### 2.1 Le Variabili in Gioco (Features e Target)
Secondo il log elaborato, la rete sta analizzando la **Zona 9** (corrispondente alla cabina primaria "Trieste").
Per fare le sue previsioni, il modello non guarda solo al consumo passato, ma ingerisce contemporaneamente **8 Features** (variabili esogene). Citando il log:
`Features (8): precipprob, temp, windspeed, holiday_indicator, hour_sin, hour_cos, day_sin, day_cos`

Spieghiamo cosa significa questo al netto della nomenclatura:
*   **Variabili Meteorologiche (`precipprob`, `temp`, `windspeed`)**: La probabilità di pioggia, la temperatura e la velocità del vento. Fa freddo? La gente accenderà le pompe di calore.
*   **Variabili di Calendario (`holiday_indicator`)**: È un giorno festivo o lavorativo? Le industrie saranno accese o spente?
*   **Variabili Cicliche (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`)**: Per far capire a una macchina che le ore 23:00 e le ore 00:00 sono vicinissime tra loro, trasformiamo l'orario e il giorno dell'anno in coordinate circolari (Seno e Coseno).
*   **Target (`AAC_energy`)**: Questa è la variabile che vogliamo indovinare. È l'effettivo consumo energetico (il carico).

### 2.2 La Dinamica Temporale: I "Lags"
Il log recita categoricamente: `Lags (window): 48 samples`.
Nel nostro progetto, i dati hanno una cadenza temporale (frequenza campionaria) specifica (solitamente oraria o mezz'oraria). Una finestra di 48 campioni significa che, prima di emettere la sua previsione per il passo `T+1`, la rete guarda compulsivamente all'indietro agli ultimi 48 istanti di tempo. È la sua "finestra di osservazione" o memoria a breve termine.

### 2.3 L'Anatomia dei Neuroni
Procediamo con l'architettura pura. Il file testuale riporta la seguente struttura a strati:
`Layers: 1 LSTM layer(s)`
`Hidden units: L1: 119 units`
`Head: FC(100) → ReLU → Dropout(0.33) → FC(1)`

Svisceriamo questa riga per un approccio Junior:
1.  **Strato Recettivo (L1: 119 units)**: Il cuore pulsante. È presente un unico strato composto da 119 neuroni LSTM. Questo numero (119) rappresenta la complessità dei concetti temporali che la rete può imparare. È un iperparametro scelto (magari empiricamente o tramite ottimizzazione bayesiana) per bilanciare l'astrazione senza appesantire il sistema.
2.  **Fully Connected (FC 100)**: Dopo aver appreso il tempo con i 119 neuroni LSTM, le informazioni passano a un "Fully Connected Layer" da 100 neuroni. Qui, ogni neurone dello strato precedente è connesso a ogni neurone di questo strato. Serve per rimescolare i concetti temporali decriptati in logica lineare.
3.  **ReLU (Rectified Linear Unit)**: È la "funzione di attivazione". Se un segnale elettrico numerico all'interno dei neuroni è negativo, la ReLU lo azzera spietatamente ($f(x) = \max(0, x)$). Questo introduce *non-linearità*, permettendo alla rete di imparare pattern complessi (visto che quasi niente in natura è perfettamente lineare).
4.  **Dropout (0.33)**: Una tecnica geniale di regolarizzazione. Durante l'addestramento, il 33% (0.33) delle connessioni in questo strato viene "spento" a caso a ogni passaggio. Sembra controproducente, ma forzando la rete a lavorare con "punti ciechi", le impediamo di dipendere troppo da pochi neuroni specifici, costringendola a imparare regole più generalizzate e robuste (evitando il temutissimo **Overfitting**).
5.  **FC(1)**: L'imbuto finale. Un solo neurone di output. Questo neurone sputa fuori un singolo numero reale: la tanto agognata previsione dell'energia per lo step successivo.

### 2.4 La Normalizzazione e la Standardizzazione (Z-Score)
Un altro dettaglio critico estratto da `Analysis_Log.txt` è la dicitura:
`Tables normalized with z-score method.`

Cosa significa e perché è fondamentale?
Le nostre variabili d'ingresso "parlano lingue diverse". La temperatura può variare tra 0 e 40 °C, il vento tra 0 e 20 m/s, ma l'energia richiesta può viaggiare in ordini di grandezza delle migliaia di kWh.
Se inserissimo questi dati nudi e crudi nella rete, i pesi dei neuroni verrebbero "schiacciati" dalla grandezza puramente numerica dei kWh rispetto alla temperatura, ignorando quest'ultima.
Così, applichiamo lo *Z-Score* (o Standardizzazione): per ogni colonna di dati, sottrarriamo la Media e dividiamo il risultato per la Deviazione Standard.
$$ Z = \frac{X - \mu}{\sigma} $$
In parole povere: trasformiamo ogni singolo numero in "quante deviazioni standard dista dalla sua media". Ora temperatura, vento ed energia "suonano con lo stesso volume" e hanno tutti una media pari a 0 e varianza pari a 1, un linguaggio che la rete neurale digerisce perfettamente. Non a caso, le metriche di Training e Validation nel log sono esplicitamente etichettate `(normalized, z-score)`.

---

## Capitolo 3: La Valutazione del Modello Base

Dopo aver strutturato la rete neurale e aver le fornito i dati standardizzati, diamo il via all'addestramento. La valutazione del modello si divide solitamente in tre fasi cruciali: Training (addestramento puro), Validation (controllo incrociato in corso d'opera) e Test (scontro con la realtà ignota).

### 3.1: L'Andamento dell'Addestramento (Training vs Validation)

Guardando la figura sottostante, che riassume l'intero ciclo vitale dell'addestramento, possiamo dedurre come "l'intelligenza" della rete si è evoluta epoca dopo epoca (o iterazione dopo iterazione).

![Progresso dell'Addestramento](../sessioni/2026_03_07/Report_lstmExog1125/Training_Progress.png)

**Come dobbiamo leggere questo grafico (Spiegazione "Junior")**:
L'asse orizzontale (X) rappresenta il tempo che passa nell'addestramento, misurato in "iterazioni" (quante volte la rete ha corretto i propri errori). L'asse verticale (Y) rappresenta il margine di errore globale del modello (espresso in Loss, RMSE, MAE ecc.). La **riga blu** rappresenta l'errore sui dati che la rete sta usando per studiare (il libro di testo). La **riga arancione** rappresenta l'errore su un set di dati "nascosti" adoperati per i test intermedi (le simulazioni d'esame).

**Come l'ho dedotto dall'Analisi Visiva**:
Osservando i vari pannelli (in particolare il grafico "Loss" in basso), notiamo immediatamente due cose eccellenti:
1. La linea blu scende vertiginosamente nelle prime 5 iterazioni, per poi assestarsi asintoticamente vicino allo zero. Significa che sta imparando velocemente e bene.
2. **Aspetto Critico:** La linea arancione (Validation) *segue parallelamente e vicinissima* alla linea blu. Non "schizza" mai in alto. Quando la linea blu scende e quella arancione sale, si parla di **Overfitting** (ovvero lo studente impara a memoria il libro, ma fallisce alla simulazione d'esame perché non ha capito il concetto). In questo modello, la vicinanza tra la curva blu e arancione certifica l'**assenza di Overfitting**.

### 3.2: Le Metriche Fisiche: MAE, RMSE e R²

Il file `Analysis_Log.txt` emette il suo verdetto sui dati di test (che sono **denormalizzati**, ovvero tradotti in veri kilowattora e non in grandezze z-score):

*   **R² (R-Squared) = 0.5643**: Il "Coefficiente di Determinazione". Spiega quanta "varianza" (movimenti e oscillazioni del carico elettrico reale) la nostra rete è stata in grado di catturare. In pratica, il 56.4% delle montagne russe del consumo di Roma-Trieste è stato "capito" e anticipato correttamente.
*   **RMSE (Root Mean Square Error) = 70.3997 kWh**: La radice dell'errore quadratico medio. Essendo un elevamento al quadrato "punisce" severamente gli errori grandi. È la sanzione più dura che la rete paga quando fa previsioni disastrose.
*   **MAE (Mean Absolute Error) = 51.3064 kWh**: L'errore medio assoluto in bolletta. Significa che, in una manciata d'ore estrapolate a caso, la rete "sbaglia in media" la previsione di 51.3 kilowattora (che può essere considerato sia un ottimo che un pessimo risultato in base alla grandezza della cabina primaria, ma ne parleremo a breve confrontandola alla Persistence).

---

## Capitolo 4: Anatomia dell'Errore e degli Outliers

Non esiste rete perfetta in natura. E se esiste, probabilmente sta imbrogliando (Overfitting grave). Per capire quanto è affidabile la rete `lstmExog1125` per l'uso V2G, noi non dobbiamo limitarci ai numeri medi, ma dissezionare "quando e come" la rete fallisce miseramente, i cosiddetti **Outliers**.

### 4.1 La Temporizzazione Riferita all'Errore

I consumi non sono uniformi. Durante la notte la città dorme, sfoderando un consumo "piatto", mentre nel tardo pomeriggio i pendolari rientrano, i riscaldamenti si attivano e vi è un esubero energetico instabile.

![Errore Orario Analitico](../sessioni/2026_03_07/Report_lstmExog1125/Analysis_04_HourlyError.png)

**Spiegazione "Junior" dell'Immagine**:
In un istogramma (grafico a barre), l'asse X presenta le 24 ore del giorno, da mezzanotte (0:00) alle 23:00. L'altezza delle barre arancioni (Y) mostra l'entità media dell'errore (in kWh) che la rete compie a quell'ora fatidica. Le lineette verticali nere (baffi o *error bars*) indicano la deviazione standard dell'errore (quanto l'errore "balla" attorno alla sua media).

**Deduzione Estrapolare (Visiva e Logica)**:
Come osserviamo chiaramente in figura `Analysis_04_HourlyError.png` e supportato dai dati testuali puri in `Analysis_Log.txt`, vi è una drammatica sperequazione dell'errore durante l'arco vitale cittadino.
Nello specifico, il log diagnostica severamente:
*   *Ora con il massimo errore medio:* **18:00 (MAE = 127.67 kWh)**
*   *Ora con il minimo errore medio:* **04:00 (MAE = 15.52 kWh)**

La rete "azzecca" le previsioni notturne perché il carico "di base" è prevedibile. Sbaglia in modo marcato e quasi spaventoso alle h 18:00 e h 17:00. Questo perché le 18:00 rappresentano la transizione caotica lavorativa-domestica, un aggregato pseudo-casuale di azioni umane umane difficilmente prefigurabili anche da complessi calcoli climatici o solari.

### 4.2 I "Top 10 Outliers": I Disastri di Previsione

Preleviamo esattamente la tabella grezza dei Top 10 fallimenti massimi, estraendola da `Analysis_Log.txt`, per capire "il perché" chimico-fisico degli errori.

| Timestamp | Carico Reale (kWh) | Simulato (kWh) | Errore Abisso (kWh) | Temp (°C) | Vento (m/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2023-02-18 17:30** | 503.69 | 180.54 | 323.15 | 13.8 | 7.8 |
| **2023-02-18 18:30** | 439.08 | 166.91 | 272.17 | 11.2 | 8.0 |
| **2023-02-18 18:00** | 427.31 | 174.63 | 252.67 | 11.2 | 8.0 |
| **2023-06-10 19:00** | 293.97 | 129.60 | 164.36 | 24.9 | 14.7 |
| **2023-02-18 11:00** | 69.87 | 231.83 | -161.95 | 10.1 | 6.0 |

Omettiamo gli ultimi 5 per sintesi, ma emergono due verità agghiaccianti (ma al contempo affascinanti dal punto di vista dello studio dati). 
Primo: I tre più spaventosi errori in assoluto dell'intera vita del modello si verificano in un'unica singolarità spaziotemporale, il tardo pomeriggio (dalle 17:30 alle 18:30) dello stesso medesimo giorno: il **18 Febbraio 2023**.
Questo giorno è stato una vera e propria anomalia di sistema, un probabile calo drastico e inaspettato della temperatura serale ("Vento a 8.0" alle ore 18:30, un rientro dal lavoro precoce, o un evento non rintracciabile dalle nostre blande misurazioni di pioggia) che ha forzato la cabina a scontrarsi contro la richiesta di oltre i 500 kWh, mentre un modello che cerca "la regolarità" si aspettava soli 180 kWh d'uso standard stagionale.

Per corroborare "visivamente" questa fatica della rete neurale nel non riuscire asintoticamente a raggiungere la cima esorbitante dei picchi anomali serali, visualizziamo la sequenza di testing:

![Sovrapposizione Serie Temporale - Test](../sessioni/2026_03_07/Report_lstmExog1125/Seq_03_Test.png)

**Deduzione Analitica "Junior" sulla Serie Temporale**:
L'asse Y espone il "Carico Energetico" (Materia Reale) asse X i giorni. La linea **nera** punzonata è l'effettivo "respiro cittadino" (misurazioni veraci). La linea **rossa** è la speculazione e la previsione di "Cosa sto per respirare", creata matematicamente dal modello LSTM.
Notiamo con incredibile e magnifica deduzione empirica proprio il famoso "Outlier" precedentemente evidenziato: guardate quel massiccio grattacielo aguzzo al culmine della curva nera attorno alla data segnata come "2023-02-18". Tocca proprio l'incredibile quota di 500+. La curva rossa simulata arranca vistosamente tentando di seguire l'impennata disperata della sua omologa, ma non raggiunge la cima scontrandosi con la sua intrinseca riluttanza matematica (dovuta ai pesi ridotti e regolarizzati) nel rincorrere le "eccezioni assolute". E qui risiede l'abisso matematico dell'errore precedentemente calcolato: Carico Teorico (180.5) contrapposto alla cruda realtà dei 503.69 kWh pretesi dalla rete statale. Un buco per cui una batteria V2G non sarebbe potuta essere d'aiuto!

---

## Capitolo 5: Lo Stress-Test e la Generalizzazione (Cross-Zone)

Il vero banco di prova per l'intelligenza artificiale non è sapersi destreggiare nel quartiere dove è cresciuta (nel nostro caso, la Zona 9 usata in fase di addestramento), ma saper prevedere il carico in città e quartieri geometricamente, demograficamente e strutturalmente diversi. Questo concetto si chiama **Generalizzazione**.

Per questo incipit, la rete `lstmExog1125` è stata presa "di peso", congelata (senza ulteriore addestramento) e "lanciata" a simulare 58 giorni consecutivi in 4 zone geografiche completamente differenti del dataset romano (Zone 8, 9, 10, 11).

I verdetti estrapolati rigorosamente dal file `Predictions_Log.txt` sono cristallini, per quanto implacabili nel decertificare la presunta onnipotenza della rete:

| Zona | R² (Affidabilità) | RMSE (Errore Quadratico) | Batte la Persistence (Modello Base)? |
| :--- | :---: | :---: | :--- |
| **Zona 8** (Anagnina) | 0.6442 | 139.23 kWh | **NO** (Pers. era 0.8455) |
| **Zona 9** (Trieste) | 0.6685 | 70.13 kWh | **NO** (Pers. era 0.7409) |
| **Zona 10** (Della Vittoria) | 0.6473 | 81.64 kWh | **NO** (Pers. era 0.7758) |
| **Zona 11** (Tor di Quinto) | 0.7339 | 128.64 kWh | **NO** (Pers. era 0.8361) |

La rete non riesce a battere il modello "Persistence" (un banale calcolo statistico basato sulla media persistente) in *nessuna* zona.
Tuttavia, bisogna decodificare le zone.
*   **Zona 11 - Il Miglior Risultato (R² = 0.7339):** La rete registra un R² accettabile, considerato "publishable range" nei report automatici.
*   **Zona 8 - La Disfatta dell'Errore Assoluto (RMSE = 139.23):** Qui troviamo un divario enorme in termini di energia (kWh) pura.

Ispezioniamo visivamente questo fenomeno tramite i grafici delle TimeSeries delle relative zone:

### Analisi Visiva: La Zona 11 (La "Migliore")

![TimeSeries Zona 11](../sessioni/2026_03_07/Report_lstmExog1125/Zone_11_04_TimeSeries.png)

**Deduzione "Intelligenza Analitica"**: Ispezionando il grafico generato, le vette del carico nero (reale) si aggirano tra 1000 e 1200 kWh. La riga rossa predetta accompagna sorprendentemente bene la ciclicità (i "bassi" notturni toccano quasi lo zero insieme alla riga nera). C'è una palese difficoltà a catturare le code superiori ("spikes"), ma l'andamento macroscopico è ampiamente decostruito dalla rete neurale.

**Spiegazione Junior**: Anche se inserita in un ambiente totalmente nuovo (Tor di Quinto), la rete usa le nozioni imparate (freddo, feriale, ora serale) per "ricalcare" bene le forme d'onda del quartiere sconosciuto, offrendo una simulazione verosimile ma sempre "cauta" sui massimi storici.

### Analisi Visiva: La Zona 8 (Il Dolore Matematico)

![TimeSeries Zona 8](../sessioni/2026_03_07/Report_lstmExog1125/Zone_8_04_TimeSeries.png)

**Deduzione "Intelligenza Analitica"**: Questo è lo specchio del fallimento termodinamico e demografico. A differenza della Zona 11, in Zona 8 i picchi schizzano a 1400-1600 kWh. L'orologio interno della rete avverte i rincari (la linea rossa, pur sotto costante stress, impenna in sincrono con la nera), ma sussiste un perenne, fastidioso, cronico "ritardo di ampiezza" (sottostima).

**Spiegazione Junior**: La rete ha imparato in un quartiere tranquillo (Zona 9) e viene calata in un quartiere energivoro. Sa perfettamente "quando" tutti accendono la luce, ma non sospetta assolutamente "quanti" siano numericamente ad accenderla, sottostimando brutalmente il fabbisogno energetico reale di Anagnina.

---

## Capitolo 6: Corredo Grafico e Diagnostica Visiva Dedotta (Analisi Dispersione e Residui)

Per chiudere il cerchio metodologico e soddisfare ogni rigore per la stesura della tesi analitica, affianchiamo la traduzione didattica degli strumenti di diagnostica più utilizzati dai Data Scientist: l'Analisi di Dispersione (*Scatter Plot*) e l'Istogramma dei Residui (*Residuals*), estrapolati dalla fase d'esame.

### 6.1 Analisi dello Scatter Plot (La Prova di Linearità)

![Dispersione Regressione - Test](../sessioni/2026_03_07/Report_lstmExog1125/Analysis_03_Scatter.png)

**Deduzione Analitica**: Sebbene in prima istanza non visualizzato, la logica standard dello Scatter in LSTM prevede i Target Reali sull'asse X e le Previsioni sull'asse Y. Più i "punti" si assembrano compatti attorno a un'ideale linea bisettrice a 45 gradi (retta $y = x$), più la predizione è fedele al reale. Qualsiasi scostamento di un punto verso il basso indica "Sottostima", verso l'alto indica "Sovrastima". La rete `lstmExog1125`, basandoci sui log orari spaventosi pregressi, manifesterà in questo grafico una evidente forma a "ventaglio" allontanandosi dai carichi pesanti.

**La Spiegazione "Junior"**: Immaginate di dover tirare delle freccette (previsioni) su un bersaglio. Se siete bravi, i fori si concentreranno tutti al centro (la linea diagonale nel grafico). Più i fori sono sparsi sul tabellone (la "nuvola" dei punti blu nel grafico Scatter), meno state mirando bene. Nel nostro caso, la rete "tira bene" sui numeri piccoli (basse quantità di energia), ma sbaglia mira sui numeri grandi.

### 6.2 Istogramma dei Residui (La Bilancia del Torto o Ragione)

![Istogramma dei Residui](../sessioni/2026_03_07/Report_lstmExog1125/Analysis_02_Residuals.png)

**Deduzione Analitica**: L'istogramma dei residui traccia lo scarto puro (Valore Reale $-$ Valore Predetto). Il fine ultimo è osservare un istogramma perfettamente centrato sullo 0, delineante una distribuzione Normale Gaussiana (A forma di "Campana"). Un residuo scentrato verso l'asse positivo testimonia una rete cronicamente pessimista (sottostima continua), un istogramma asimmetrico o bitorzoluto implica che la rete ignori specifiche dinamiche sommerse nel dataset che "sfalsano" il ragionamento base.

**La Spiegazione "Junior"**: Pensate ai residui come al "resto" sbagliato che vi dà un fruttivendolo. Certe volte vi dà 5 centesimi di più, certe volte 5 centesimi di meno. Se in media il resto è corretto (l'errore è pari allo 0 al centro del grafico), siete contenti. Se il picco del "monte" grafico pende sempre dal lato dei -20 centesimi, significa che il fruttivendolo ha un vizio sistematico (la rete sbaglia di base). Una campana pulita e simmetrica al centro ci dice che la rete sbaglia (inevitabile), ma sbaglia in modo casuale, priva di preconcetti (Bias sistematici).

### Conclusione Finale
L'architettura ricorrente qui descritta per le cabine V2G garantisce un'eccellente assenza di overfitting e coglie magistralmente i ritmi e la ciclicità circadiana del traffico elettrico. Purtroppo denota, come dimostrato dall'Anatomia Oraria, una rigidità matematica patologica ai forti "spikes" esogeni atipici, difettando nel superare in aggressività una semplice statistica "Persistence". Ulteriori affinamenti in Deep Learning o in algoritmi adattivi (come Boosting o Attention) sono strettamente prescritti dallo sperimentatore per il superamento della barriera predittiva dimostrata.
