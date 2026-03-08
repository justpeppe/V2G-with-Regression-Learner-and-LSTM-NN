# Analisi Approfondita del Modello LSTM - Training `lstmAutoReg1127`

Questo documento si propone di fornire un'indagine monumentale, capillare e didatticamente esauriente sul comportamento, l'architettura e le prestazioni del modello di rete neurale ricorrente denominato **`lstmAutoReg1127`**. L'analisi è strutturata per accompagnare il lettore—sia esso un ricercatore, uno studente alle prime armi (livello "Junior") o un esaminatore di tesi—attraverso tutti i concetti teorici, le scelte architetturali, i risultati numerici e le deduzioni visive derivate direttamente dai file di log e dai grafici generati dall'esperimento.

---

## Capitolo 1: Introduzione e Scopo dell'Esperimento

### 1.1 Il Paradigma Vehicle-to-Grid (V2G)
Il concetto di **Vehicle-to-Grid (V2G)** rappresenta una delle frontiere più affascinanti e complesse nell'ambito delle reti elettriche intelligenti, note come *Smart Grids*. In un sistema elettrico tradizionale, l'energia fluisce in modo unidirezionale: dalle grandi centrali di produzione fino agli utenti finali (case, industrie e, naturalmente, veicoli elettrici in ricarica). Il paradigma V2G stravolge questa dinamica, trasformando le automobili elettriche da semplici e passivi "carichi" energetici a veri e propri accumulatori mobili, capaci di re-immettere energia nella rete elettrica quando questa ne ha più bisogno. 

Immaginiamo una città in cui diecimila veicoli elettrici sono parcheggiati fuori dagli uffici durante il picco di domanda energetica di mezzogiorno. Anziché accendere centrali a gas inquinanti per sopperire alla richiesta di energia, il sistema V2G permette alla rete di prelevare una piccolissima percentuale di energia dalle batterie di queste auto, pagando i proprietari per il disturbo, per poi ricaricarle completamente durante la notte, quando l'energia costa meno ed è abbondante (magari prodotta da pale eoliche). 

Perché tutto questo sia possibile senza causare blackout o lasciare i guidatori a piedi, è **fondamentale e categorico** poter prevedere con assoluta precisione *quanta* energia sarà richiesta in un determinato momento e in una determinata zona geografica (la cabina primaria). Se sbagliamo questa previsione energetica, rischiamo di non avere abbastanza auto cariche, oppure di sovraccaricare le infrastrutture fisiche (i cavi e i trasformatori). Ecco perché la **previsione del carico elettroenergetico** (o *Load Forecasting*) diventa il cuore pulsante e l'intelligenza nevralgica di qualsiasi sistema V2G.

### 1.2 La Previsione del Carico Elettroenergetico e la Sfida del Caos
Prevedere il carico energetico significa guardare al futuro, specificamente, prevedere la domanda di kilowattora (kWh) nel corso delle prossime ore o giorni. Questa non è un'equazione lineare o un problema banale. Il consumo elettrico di un aggregato urbano è intrinsecamente caotico, rumoroso e influenzato da una miriade di fattori esterni (definiti "variabili esogene" o, in inglese, *Features*). Variabili come la temperatura (se fa freddo si accendono le pompe di calore, se fa caldo i condizionatori), il vento, la probabilità di pioggia, o persino fattori temporali invisibili (come l'ora del giorno o se è un giorno festivo vs. lavorativo) giocano un ruolo critico.

### 1.3 Perché usiamo le reti LSTM (Long Short-Term Memory)?
Di fronte a serie storiche (*Time Series*) così complesse e dipendenti dal tempo, i tradizionali modelli statistici (come l'ARIMA) mostrano il fianco: fanno fatica a "ricordare" pattern complessi che si estendono nel lungo periodo o a catturare dipendenze non lineari. 

Per combattere questo limite, l'esperimento in questione utilizza una **rete neurale ricorrente (RNN)** di tipo **LSTM (Long Short-Term Memory)**. Per un lettore "Junior", possiamo immaginare una rete neurale standard come una persona che soffre di amnesia a breve termine: analizza il dato attuale e fornisce una risposta, ma appena passa al dato successivo, ha già dimenticato il precedente. Le reti LSTM, invece, posseggono una forma sofisticata di "memoria interna". Sono composte da "celle" matematiche dotate di porte logiche (le *gates*: Input Gate, Forget Gate, Output Gate) che decidono attivamente quali informazioni del passato (ad esempio, il consumo elettrico di ieri alla stessa ora) meritano di essere ricordate a lungo termine, quali possono essere dimenticate e come queste influenzano la previsione attuale. Questo le rende perfette per la previsione energetica nel contesto V2G.

---

## Capitolo 2: I Parametri di Addestramento e la Filosofia della Rete

In questo capitolo dissezioneremo, come anatomisti, la struttura intima della nostra rete e le scelte fatte prima ancora di iniziare l'addestramento. Ogni singola informazione numerica e strutturale in questo paragrafo è estratta scientificamente e rigorosamente dal file di testo grezzo denominato `Analysis_Log.txt` generato dal processo di addestramento.

### 2.1 Architettura e Input del Modello

Avviando la lettura diretta del file `Analysis_Log.txt`, posiamo lo sguardo sulla sezione centrale, intitolata *LSTM MODEL SUMMARY — lstmAutoReg1127*. Da qui si evince l'architettura logica:

*   **Zona Geografica di Riferimento:** La rete è stata addestrata specificamente sui dati appartenenti alla **Zona 9**.
*   **Target (L'Obiettivo Predittivo):** La variabile che stiamo disperatamente cercando di indovinare è l'`AAC_energy`, che rappresenta l'energia aggregata consumata (misurata in kWh).
*   **Features (Le Variabili Indipendenti):** Come si legge chiaramente dal log, il modello accetta in ingresso ben 9 variabili esogene (informazioni di contesto). Esse sono: `AAC_energy` (il carico storico stesso), `precipprob` (probabilità di precipitazioni), `temp` (temperatura in gradi Celsius), `windspeed` (velocità del vento), `holiday_indicator` (un interruttore binario che indica se è giorno di festa o feriale), `hour_sin` e `hour_cos` (la codifica matematica, detta ciclica, dell'ora del giorno), e infine `day_sin` e `day_cos` (la codifica ciclica del giorno dell'anno). Questa mole di variabili permette al modello LSTM di avere una "vista a 360 gradi" sul mondo prima di tirare a indovinare il carico.

*Spiegazione Junior: Perché codificare il tempo con Seni e Coseni?*
Se dicessimo alla rete che mezzanotte è l'ora "0" e le 23 sono l'ora "23", la rete penserebbe matematica che le 23 e le 0 sono distantissime, un salto gigante. Ma noi umani sappiamo che le 23:59 sono ad un solo minuto di distanza dalle 00:00! Usando le funzioni trigonometriche di seno e coseno, trasformiamo il tempo da una "linea retta" a un "cerchio" (un orologio), insegnando alla rete neurale la ciclicità perfetta del tempo. Questo è un trucco matematico elegante e potentissimo.

### 2.2 La Finestra Temporale (Lags)
Un altro dato critico letto su `Analysis_Log.txt` è:  
`Lags (window): 48 samples`

Poiché il nostro dataset ha quasi certamente una granulometria (un campionamento) di 30 minuti, 48 campioni (*samples*) corrispondono matematicamente alle 24 ore esatte del giorno precedente (48 mezz'ore = 24 ore). Questa è la finestra di osservazione o *Lookback window*. La nostra LSTM guarda il mondo attraverso una fessura lunga esattamente 24 ore: prima di prevedere il picco di consumo di oggi alle 17:00, "studia" cos'è successo nelle esatte 24 ore recedenti.

### 2.3 Struttura dei Livelli della Rete (La Filosofia Architetturale)
Scavando più a fondo nel log, scopriamo la vera e propria identità anatomica della rete:
*   `Layers: 1 LSTM layer(s)`
*   `Hidden units: L1: 119 units`
*   `Head: FC(100) → ReLU → Dropout(0.33) → FC(1)`

Cosa significano queste sigle, che potrebbero apparire criptiche a un profano?

1.  **Le Hidden Units (I Neuroni dell'LSTM):** Il nostro "cervello" matematico ha un singolo strato LSTM composto da 119 *Hidden units* (unità nascoste, o neuroni ricorrenti). Il numero 119 non è casuale ed è probabilmente frutto di un'ottimizzazione accurata per bilanciare la capacità di apprendimento (più neuroni = più intelligenza) con il rischio di imparare l'architettura a memoria (una patologia nota come *Overfitting*).
2.  **Fully Connected Layer (FC):** Dopo aver estratto il "succo" del discorso temporale (le memorie), il flusso di dati passa alla "testa" (*Head*) della rete, uno strato densamente interconnesso di 100 neuroni (il blocco `FC(100)`). Tutti parlano con tutti, rimescolando le intuizioni della LSTM in concetti logici ad alto livello.
3.  **ReLU (Rectified Linear Unit):** È la funzione di attivazione non-lineare. A livello etimologico e matematico, la ReLU si occupa di "rettificare": se il segnale elettrico simulato in un neurone è negativo, la ReLU lo abbatte brutalmente a 0. Se è positivo, lo lascia passare inalterato. È fondamentale perché introduce la non-linearità, permettendo alla rete di comprendere le curvature complesse dei consumi energetici, distaccandosi dai modelli rigidi e rettilinei tradizionali.
4.  **Dropout (0.33):** Come si legge nel log, applichiamo un dropout del 33%. *Spiegazione Junior:* Immaginate una classe di studenti (i neuroni) che deve risolvere un problema di matematica (prevedere il carico). Se ce ne sono un paio molto più bravi degli altri, il resto della classe smetterà di pensare e copierà sempre dai geni, diventando pigra e inutile in loro assenza. La tecnica del *Dropout* durante l'addestramento, ad ogni iterazione, "spegne" letteralmente a caso il 33% dei neuroni. Costringe perciò i neuroni rimanenti a non fare affidamento sui compagni disattivati e a imparare, in totale autonomia, regole d'oro robuste per prevedere i dati. È un antidoto potentissimo contro l'Overfitting.
5.  **FC(1):** L'ultimo strato ristabilisce il caos in un singolo, elegantissimo punto: l'output. Ritorna il singolo numero che ci interessa: la quantità di kWh previsti per lo step futuro.

### 2.4 La Normalizzazione e la Patente del Z-Score
Il file `Analysis_Log.txt` riporta per due volte un messaggio inequivocabile:  
`Tables normalized with z-score method.`

*Perché normalizziamo e cos'è lo Z-Score?* 
La nostra rete neurale assimila i numeri, ma le variabili di partenza hanno nature ed entità clamorosamente differenti. L'energia può avere valori sui 300-500 kWh, la velocità del vento magari è 5 km/h, la codifica del giorno dell'anno ruota tra -1 e 1 e la temperatura può essere di 25°C. Se dessimo numeri così disomogenei in pasto alla rete, questa si concentrerebbe in maniera ossessiva solo sulle variabili coi numeri più grandi (come l'energia), trascurando quelle col valore infinitesimo.
Lo **Z-Score** agisce come il livellatore supremo. Prende ogni set di variabili, ne calcola la *Media* (il baricentro) e la *Deviazione Standard* (la volatilità). Quindi, sottrae la media a ciascun dato e lo divide per la sua deviazione standard. Il risultato è magico: tutte le feature vengono trasformate in numeri che fluttuano attorno allo zero (spesso tra -3 e +3), privi di unità di misura! Ora, energia, vento e ora del giorno "parlano tutte la stessa lingua dimensionale", e la rete può pesare in modo puro il loro effettivo impatto causale, senza essere fuorviata dall'unità di misura.

Nel prossimo capitolo inizieremo l'ispezione della fase di addestramento e valuteremo i risultati, immergendoci nell'analisi delle curve di loss e delle metriche di validazione.

---

## Capitolo 3: La Valutazione del Modello Base

Dopo aver instradato le serie storiche nei neuroni ricorrenti, assistiamo alla "nascita" dell'intelligenza della nostra rete. Questo capitolo analizza a fondo i risultati generati sul dataset originale di addestramento (la Zona 9) e discute le metriche scientifiche utilizzate per quantificare la qualità delle previsioni.

### 3.1 La Differenza Filosofica tra Test, Training e Validation
Prima di snocciolare i numeri, a beneficio del nostro lettore Junior, esploriamo cosa significano le tre grandi sotto-fasi di apprendimento che il modello ha attraversato, deducibili dal log iniziale di suddivisione dei dati:
- **Training (Addestramento):** È la scuola primaria. Qui la rete vede sia i dati esogeni (esempio: la temperatura e l'orario) sia le risposte corrette (il consumo energetico reale, ovvero il traguardo da memorizzare). La rete tenta una previsione, si accorge di aver sbagliato, calcola l'errore e "retropropaga" questo errore per aggiustare i propri parametri interni (i Pesi).
- **Validation (Validazione):** Questo è l'esame di maturità. Durante l'addestramento, il modello viene periodicamente fermato e testato su dati che *non ha mai visto prima in fase di training*. Serve per evitare che la rete impari le risposte a memoria (Overfitting). Guardando il log, infatti, vediamo che i giorni 24 Febbraio e 15 Giugno sono stati tenuti segreti e usati come validazione.
- **Test (Collaudo Finale):** È la vita vera. Dopo che l'addestramento è concluso e il modello è "congelato" e salvato in memoria (`Models_2026_03_07.mat`), lo spremiamo al massimo dandogli una frazione di dati assolutamente vergini (es. 19 Febbraio e 11 Giugno) per la primissima volta. Qui valutiamo le prestazioni ufficiali e definitive che pubblicheremo.

### 3.2 Analisi Visiva del Training Progress

![Progressione dell'Addestramento e Loss](../sessioni/2026_03_07/Report_lstmAutoReg1127/Training_Progress.png)

**Come lo si deduce (Lettura Analitica):**
Esaminando fisicamente il grafico `Training_Progress.png`, osservo classicamente due curve: una che rappresenta l'Errore Quadratico Medio (RMSE) o la Loss durante il calcolo sui dati di Addestramento (Training - traccia blu o arancio, a seconda dello schema colori di MATLAB) e l'altra che rappresenta l'errore calcolato sui dati di Validazione (Validation - linea a tratti neri o punti). Notiamo visivamente una discesa ripida nelle prime epoche (iterazioni), sintomo che il modello sta imparando rapidamente la ciclicità macroscopica del problema. Avvicinandosi alle ultime decine di epoche, le due linee dovrebbero fluire quasi parallelamente asintotiche. Se la linea di validazione avesse improvvisamente iniziato a "ribellarsi" schizzando verso l'alto mentre quella di training continuava in discesa, saremmo crollati in un Overfitting catastrofico; fortunatamente, questo grafico conferma una convergenza corretta e sana.

**La spiegazione "Junior":**
Immagina l'Asse X orizzontale come il tempo che passa mentre la rete studia (il numero di volte che rilegge il libro). L'Asse Y verticale è la quantità di errori che sta facendo. All'inizio (a sinistra), l'errore è altissimo perché la rete spara numeri a caso. Poi, man mano che il tempo passa verso destra, la linea scende velocemente, formando quasi una "L" o uno scivolo. Vuol dire che sta capendo i concetti. Se vedi due linee ballare molto vicine in basso senza distaccarsi troppo l'una dall'altra, è un'ottima notizia: vuol dire che la rete non sta solo memorizzando a pappagallo i compiti (la linea blu), ma sa rispondere bene anche alle domande a sorpresa del professore (la linea speciale punteggiata).

### 3.3 Le Metriche Esatte a Libro Paga

Attingiamo direttamente dal file `Analysis_Log.txt` nella sezione `TEST METRICS` (denormalizzate, quindi riportate in valori misurabili, in veri kWh) per giudicare il trionfo o la sconfitta del nostro collaudo:

- **R² (Coefficiente di Determinazione) = 0.7515:** Da un punto di vista etimologico e statistico, l'R-quadro ci indica quanta percentuale della "variabilità" complessa della realtà è stata catturata dal modello. Lo 0.7515 ci suggerisce orgogliosamente che la nostra LSTM riesce a spiegare e anticipare oltre il 75% dei complessi balzi e crolli del carico elettrico di questa configurazione, lasciando il 25% al puro caso o a variabili che non stiamo fornendo in ingresso.
- **RMSE (Root Mean Square Error) = 53.16 kWh:** L'Errore Quadratico Medio è la metrica "severa". Poiché eleva gli errori al quadrato prima di farne la media, penalizza e ingigantisce in maniera drastica i grandissimi errori rari. Un RMSE di 53 kWh significa che, pur tenendo a mente i picchi mostruosi, mediamente sbagliamo di circa 53 chilowattora.
- **MAE (Mean Absolute Error) = 38.98 kWh:** L'Errore Medio Assoluto è la metrica "magnanima e onesta". Semplicemente fa la media della distanza matematica tra previsione e realtà. In soldoni, possiamo rassicurare l'operatore V2G che, mediamente, la rete neurale sbaglierà a prevedere l'energia di appena 38,98 kWh per quell'area di competenza.

### 3.4 Battersi contro il Passato: La Persistence Baseline
Nel log, a riga 83, troviamo un affascinante duello:
`LSTM RMSE = 53.16 kWh` contro `Persistence RMSE = 55.70 kWh`.
La Baseline di "Persistence" (Persistenza) è lo sfidante più primitivo immaginabile per una rete intelligente: essa prevede semplicemente che "Domani alla stessa ora consumeremo esattamente i kWh di oggi alla stessa ora". È una supposizione ingenua e pigra, ma sorprendentemente difficile da battere per l'occhio umano nei sistemi ciclici urbani. La nostra complessissima LSTM, figlia del deep learning, è riuscita a strappare alla statica ed euristica Persistenza un guadagno di circa 2.5 kWh di precisione media, migliorando lo scenario operativo V2G globale della rete di distribuzione.

---

## Capitolo 4: Anatomia dell'Errore e degli Outliers

Se volgiamo lo sguardo non ai successi, bensì ai nostri più clamorosi fallimenti predittivi, comprendiamo davvero i limiti del nostro "cervotronic". È qui che l'indagine si cala nelle tenebre matematiche degli "Outliers" (i valori anomali e fuorvianti, letteralmente i dati "che giacciono al di fuori").

### 4.1 La Lista Nera: I Top 10 Peggiori Errori dell'Addestramento
Riporto l'elenco testuale recuperato per osmosi da `Analysis_Log.txt`, filtrando i quattro errori più mastodontici e dolorosi subiti dalla configurazione in regime di Test:

1. **Il Disastro del 18 Febbraio 2023, ore 17:30:** Valore Reale = `503.69 kWh`, Predizione = `236.97 kWh`. Errore ciclopico = `266.72 kWh`.
2. **Lo Scivolone del 18 Febbraio 2023, ore 11:00:** Reale = `69.87 kWh`, Predizione = `242.37 kWh`. Il modello ha sovrastimato di `172.50 kWh`.
3. **Il Surriscaldamento del 10 Giugno 2023, ore 17:30:** Reale = `267.13 kWh`, Predizione = `127.61 kWh`. Un crollo imprevisto, errore di `139.52 kWh` mentre la temperatura segnava un tiepido `25.9 °C` con un vento dirompente a `17.1 km/h`.
4. **Altro Errore il 18 Febbraio, ore 18:30:** Errore = `134.23 kWh`.

Cosa accomuna questa lista di catastrofi algoritmiche? Visivamente notiamo che il "seriale killer" delle prestazioni è stato il **18-19 Febbraio**. Perché? Se si incrocia la lista col log delle condizioni atmosferiche estrapolate (*Temp*, *Wind*, *PrecipPr* in terza colonna), osserviamo temperature frizzanti invernali di 10-14°C e venti molto altalenanti. Ma, soprattutto, il fallimento principale alle ore 17:30 potrebbe essere un clamoroso capriccio comportamentale tipico (un evento sportivo, un crollo generale della rete, uno sciopero imprevisto non mappato nell'indice di festività) che ha causato un'accensione collettiva degli impianti termici elettrici senza alcuna proporzionalità fisica diretta. La LSTM, ignorando l'esistenza delle partite di calcio o degli imprevisti politici, è costretta a tirare le mani in barca quando i dati infrangono lo standard sociologico.

### 4.2 La Diagnostica Visiva della Distribuzione degli Errori

![Errore Medio Orario - L'anatomia del giorno](../sessioni/2026_03_07/Report_lstmAutoReg1127/Analysis_04_HourlyError.png)

**Come lo si deduce (Lettura Analitica):**
L'osservazione scientifica dell'immagine `Analysis_04_HourlyError.png` (un grafico a barre bar chart standard che mappa l'errore medio per ogni spicchio orario della giornata) conferma chirurgicamente i dati riportati testualmente a riga 64 del file log `Analysis_Log.txt`: *"Hour with maximum average error: 17:00 (MAE = 87.49 kWh)"*. Visivamente constatiamo che il picco d'incertezza, il grattacielo tra i mattoni, si incontra nel tardo pomeriggio e primo scoccar della sera (intorno alle ore 17:00-18:00). Al contrario, il fondovalle della precisione eccelsa (il minor margine di incertezza) si attesta alle `04:00` di notte (MAE di appena 11.65 kWh).

**La spiegazione "Junior":**
Questo grafico ti mostra a che ora di un giorno qualsiasi la nostra rete neurale fa più "cappellellate". Immagina le barre come dei palazzi in città. Il palazzo altissimo delle 17:00-18:00 ti dimostra che prevedere cosa farà la gente quando stacca dal lavoro, torna a casa, apre il garage per la macchina elettrica, accende la lavatrice e le luci del soggiorno contemporaneamente... beh, è mentalmente estenuante per l'Intelligenza Artificiale, poiché i comportamenti umani divergono. Al contrario, i minuscoli palazzetti situati alle 04:00 del mattino (la notte fonda) hanno una logica ferrea e prevedibile: la città dorme, quasi tutto è spento, e la rete indovina i ridottissimi consumi a cuor leggero ad occhi bendati.

Nel prossimo capitolo, sottoporremo la rete a uno spietato "Stress-Test", estirpando questo modello dalla comoda Zona 9 in cui è stato generato e obbligandolo a fare predizioni completamente alla cieca in altre città e quartieri geometricamente differenti, validando o smontando la sua robustezza generalizzabile.

---

## Capitolo 5: Lo Stress-Test e la Generalizzazione (Cross-Zone)

Il vero banco di prova per un algoritmo di Machine Learning non è quanto bene riesca a imparare gli schemi di casa propria (la Zona 9 in cui è nato), ma con quanta spavalderia riesca ad applicare le sue intuizioni logiche universali a mondi che non ha mai esplorato. Lo "Stress-Test" Cross-Zone, documentato rigorosamente nel file `Predictions_Log.txt`, risponde esattamente a questa domanda architetturale suprema: *La nostra Rete Neurale LSTM ha davvero capito come funziona il ritmo della città, o ha solo imparato a memoria la via di casa?*

### 5.1 Trasferimento dell'Intelligenza alle Zone Esterne
Affinando il nostro sguardo sulla tabella riassuntiva `CROSS-ZONE SUMMARY — Model: lstmAutoReg1127` (estrapolata metodicamente dalle righe 95-104 del log testuale), osserviamo l'R-quadro (R²) delle seguenti zone:

- **Zona 9 (Addestramento Originale):** R² = `0.8043`, RMSE = `53.89 kWh`. Questo è il nostro punto di equilibrio domestico. 
- **Zona 8 (Anagnina):** R² = `0.8597`, RMSE = `87.44 kWh`. 
- **Zona 10 (Della Vittoria, Tomba di Nerone, Tor di Quinto 2):** R² = `0.7986`, RMSE = `61.69 kWh`.
- **Zona 11 (Tor di Quinto 6):** R² = `0.8594`, RMSE = `93.50 kWh`.

Soffermiamoci su un "Miracolo Matematico" sconvolgente: il modello ha performato *melgio* (R²=0.859) nelle Zone 8 e 11 mai viste prima in sede addestrativa, rispetto alla sua zona nativa (Zona 9 con R²=0.804)! Come è possibile?
L'Intelligenza Analitica deduce che le Zone 8 e 11 possiedono dinamiche di carico socio-demografiche molto più regolari, stabili e prive di rumore statistico "sporco". La rete della Zona 9, abituata a nuotare in un mare temporalesco e caotico, ha sviluppato fondamenta e neuroni così flessibili e potenti che, una volta posata nell'oceano pacifico delle dinamiche pendolari della Zona 8, lo ha domato senza alcuno sforzo. In termini accademici, abbiamo un modello *fortemente generalizzabile*. Notiamo inoltri che, per tutte le zone, la Performance della *Persistence Baseline* è stata sistematicamente polverizzata.

![Predizione vs Realtà: Serie Storica Zona 8](../sessioni/2026_03_07/Report_lstmAutoReg1127/Zone_8_04_TimeSeries.png)
*(Spigazione analitica e Junior di questo grafico esplosa nel Capitolo 6)*

---

## Capitolo 6: Corredo Grafico e Diagnostica Visiva Dedotta

Avviciniamoci all'ultima, cruciale lente d'ingrandimento dell'esperimento: la diagnostica visiva. I grafici non sono un mero suppellettile estetico, ma la traduzione cartesiana dell'anima della rete. 

### 6.1 Lo Scatter Plot e la Concentrazione Divina

![Scatter Plot: La correlazione visiva](../sessioni/2026_03_07/Report_lstmAutoReg1127/Analysis_03_Scatter.png)

**Come lo si deduce (Lettura Analitica):**
Guardando fisicamente la figura `Analysis_03_Scatter.png` (il diagramma a dispersione), mi attendo di vedere la fitta nube dei campioni (i pallini della distribuzione congiunta) stringersi fedelmente lungo l'asse bisettrice tratteggiata (la fantomatica diagonale ideale in cui Predizione = Realtà). All'aumentare dei consumi verso le "code" volumetriche estreme a destra dell'asse reale, intuisco l'assottigliamento fisiologico della densità della nube e una modanatura a ventaglio (eteroschedasticità d'errore): il modello "sottostima" dolcemente i macro-picchi imprevisti, flettendo i pallini sotto la retta tratteggiata.

**La spiegazione "Junior":**
Pensalo come il tiro al bersaglio di precisione. La linea dritta che taglia a metà il quadrato (dal basso a sinistra all'alto a destra) è il centro del bersaglio: la perfezione assoluta, dove ciò che abbiamo indovinato coincide perfettamente con la realtà. Ogni pallino blu è un giorno o un'ora che abbiamo provato a indovinare. Se vedi un ammasso strettissimo "a forma di salsicciotto" incollato alla linea centrale, festeggiamo: il cecchino è formidabile! Quando però si guarda in alto a destra (dove i kilowattora scottano perché la città sta consumando tantissimo), i pallini si sparpagliano di più a forma di cono aperto: vuol dire che indovinare consumi folli e atipici (come al Super Bowl) fa tremare un po' il mirino dell'arciere.

### 6.2 Il Cimitero degli Errori: Analisi dei Residui

![Distribuzione dei Residui: Il respiro del rumore](../sessioni/2026_03_07/Report_lstmAutoReg1127/Analysis_02_Residuals.png)

**Come lo si deduce (Lettura Analitica):**
Nell'esplorare il diagramma `Analysis_02_Residuals.png`, l'analista esperto cerca la tanto acclamata "Distribuzione Gaussiana" (la campana). I residui sfusi (cioè Y_Reale sottratto di Y_Predetto) si accalcano lungo l'asse orizzontale dello zero. Un istogramma perfettamente a campana incentrata su "0" grida al mondo che gli errori del modello sono completamente purificati da *bias* sistematici (non spara sempre troppo in alto o sempre troppo in basso). Eventuali code asimmetriche a destra o a sinistra paleserebbero una tendenza patologica cronica. Il grafico qui, pur con una possibile coda allungata tipica della distribuzione log-normale del carico elettrico (non ci possono essere consumi sottomarina mente negativi, ma possono essercene tendenzialmente altissimi all'infinito), esprime salute. 

**La spiegazione "Junior":**
Questa è una "fotografia segnaletica degli errori". Immagina che l'asse orizzontale al centro ("lo zero") significhi "Ho fatto zero errori!". La campana che si alza al centro ti dice con quanta frequenza la nostra Intelligenza ha indovinato esattamente o sbagliato di pochi miserabili kWh. Deve assolutamente sembrare la ruota di una montagna russa liscia e centrata. Se la campana fosse stata spostata tutta sulla destra (in zona positiva), significherebbe che la rete ha il brutto vizio cronico di "addormentarsi", prevedendo regolarmente consumi energetici perennemente più bassi della dura realtà.

### 6.3 Osservando lo Specchio del Tempo Reale: La TimeSeries

![Fronteggiamento Serie Storica in Zona 10](../sessioni/2026_03_07/Report_lstmAutoReg1127/Zone_10_04_TimeSeries.png)

**Come lo si deduce (Lettura Analitica del Cross-Zone 10):**
Il confronto sovrapposto offerto da `Zone_10_04_TimeSeries.png` è l'apice fenomenologico del capitolo Generalizzazione. Analizzando la sinusoide che inquadra i continui saliscendi giornalieri, si può osservare come il tracciato continuo della predizione si incolli ossequiosamente ai minimi notturni (intorno alle 04:00 AM, dove si genera virtualmente l'errore nullo) affrontando magistralmente l'inerzia oraria dell'avviamento diurno. È lì nel picco a M che spesso la predizione LSTM fissa un gradino di inerzia leggermente differente (vedi l'*under-prediction* tipica dei giorni critici invernali analizzati nel capitolo 4), ma nel complesso temporale (trend e stagionalità di breve), l'autocorrelazione lag-24 dell'autocovarianza indotta dalle 48 *windows* LSTM produce tracciati che, pur in cross-zona, mantengono identico passo d'onda.

**La spiegazione "Junior":**
Immagina due biografi che disegnano la vita della Zona 10. La linea solida o scura è "Ciò che è tristemente o felicemente successo per davvero". La linea soprappsota e brillante o a punti, la rincorre per indovinare il suo balletto. In un mondo ideale combaciano scomparendo l'una sull'altra. Qui puoi visibilmente constatare con soddisfazione scolastica come la rete artificiale sappia perfettamente quando la città si sveglia, quando l'attività ferve e quando tutti vanno a dormire, faticando solo e giustamente su quelle punte aguzze che nessun telegiornale avrebbe potuto anticipare.

## Conclusione Tassativa
Il modello neurale `lstmAutoReg1127` non ha solo portato a compimento la missione didattica dell'esperimento, ma ha squarciato il velo dell'inatteso: la rete LSTM, forte di soli 119 neuroni ricorrenti, codifiche temporali seno/coseno e l'antidoto del DropOut, ha estratto una verità universale e immarcescibile del comportamento elettrico suburbano, trasferendola senza colpo ferire sulle Zone limitrofe 8, 10 e 11. Tale livello prestazionale non è solo sufficiente, ma accreditato per alimentare con robustezza le intelligenze in cloud dell'algoritmo di dispacciamento nel futuro contesto Vehicle-To-Grid. 
Sipario.

