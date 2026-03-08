---
name: lstm-report-junior-thesis-analyzer
description: Analizza una cartella Report_lstm di output (contenente log e grafici) e genera un'analisi iper-dettagliata e prolissa, adatta per la stesura di una tesi di livello Junior. Spiega ogni termine tecnico, ogni assunzione, cita le fonti, e inserisce ed analizza analiticamente le immagini generate direttamente nel testo.
---

# Scopo della Skill
Questa skill deve essere invocata quando l'utente desidera un'analisi monumentale, esaustiva, ridondante a scopi didattici e di tesi per un preciso addestramento LSTM del proprio progetto Vehicle-to-Grid (V2G).

**Obiettivo dimensionale:** Sii il più esplicito e descrittivo possibile. Non c'è alcun limite di caratteri; è auspicabile e probabile superare ampiamente i 30.000 caratteri totali per garantire la massima profondità. Spiega tutto a fondo senza restrizioni di lunghezza.
**Target dell'analisi:** Un lettore "Junior" (studente alle prime armi, professore non del campo o esaminatore di tesi) a cui va spiegato *ogni singolo passaggio matematico, statistico e concettuale*. Devono rigorosamente essere citate le fonti da cui si prelevano le interpretazioni numeriche o visive, allegando le relative figure.

## Flusso di Esecuzione (Istruzioni per l'Agente)

1.  **Esplorazione Completa della Cartella:** L'utente fornirà il percorso di una cartella `Report_lstm.../`. Come prima cosa, usa un tool per listare interamente la directory così da **stampare TUTTI i file** presenti nella cartella senza saltarne nessuno (comprese immagini, log specifici di zona, ecc.).
2.  **Piano di Analisi e Approvazione Utente:** Sulla base dei file trovati al punto precedente, stila e restituisci all'utente un piano dettagliato per analizzarli in ordine logico. **CRITICO: FERMATI qui e richiedi esplicitamente l'approvazione dell'utente** (usando `notify_user` o fermando la chain) prima di iniziare a scrivere effettivamente il report.
3.  **Lettura e Ispezione dei File:** Ricevuta l'approvazione, procedi con la lettura di tutti i file log e l'apertura delle immagini. **CRITICO: Devi APRIRE effettivamente i grafici principali** (es. con il tool `view_file` sui file `.png`) in modo da guardarli veramente per non tirare a indovinare!
4.  **Salvataggio e Nomenclatura del File:** Dai log, individua il nome del modello addestrato (es. `lstmExog1125`). Il documento finale in Markdown generato dovrà obbligatoriamente chiamarsi `analisiTesi[NomeReteCamelCase].md` all'interno della cartella `docs/` (es. `docs/analisiTesiLstmExog1125.md`).
5.  **Identificazione Meta-Informazioni, Tabelle e Citazioni Dirette:** Dal log identifica le zone predette, l'orario dei picchi di errore, gli R², l'RMSE, l'architettura della rete, la baseline Persistence e i Top 10 Outliers.
    *   **CRITICO (Fonti Dirette):** Ad ogni affermazione, dato numerico o deduzione, **devi esplicitamente e rigorosamente citare la fonte diretta** (il file testuale o l'immagine) da cui l'hai prelevata (es. *"Come si evince dall'analisi diretta del file `Analysis_Log.txt`...*" oppure *"Come possiamo osservare chiaramente in figura `Seq_03_Test.png`, la curva..."*).
    *   **Strutturazione e Tabelle:** Utilizza attivamente i dati grezzi estratti dai file `.txt` e le informazioni visive delle immagini per costruire **tabelle riassuntive markdown**, elenchi puntati strutturati e box esplicativi. Il documento non deve essere solo un muro di testo, ma un elaborato estremamente **ben organizzato, formattato in modo accademico e facilmente consultabile**.
6.  **Elaborazione del Contenuto (Il Documento di Tesi) - Scrittura Modulare:**
    Il report generato DEVE essere suddiviso nei seguenti grandi capitoli. **CRITICO: Devi elaborare, scrivere e salvare nel file UN CAPITOLO ALLA VOLTA, utilizzando chiamate iterative o strumenti di edit incrementali per evitare blocchi o bottleneck testuali dovuti ai limiti di output.** Sii estremamente prolisso, usa ampiamente le metafore, sii estremamente ridondante, rinfresca la memoria del lettore più volte e inframezza ciascun capitolo con le **immagini pertinenti**. L'obiettivo è analizzare a fondo tutto.

    *   **Capitolo 1: Introduzione e Scopo dell'Esperimento.** Spiega cos'è il V2G, cos'è la previsione del carico elettroenergetico, perché LSTM.
    *   **Capitolo 2: I Parametri di Addestramento e la Filosofia della Rete.** Preleva i dati da `Analysis_Log.txt`. Citalo rigorosamente. Spiega Neurone, Dropout, Z-Score. 
    *   **Capitolo 3: La Valutazione del Modello Base.** Inserisci il grafico `Training_Progress.png` (usando sintassi `![Didascalia](percorso/assoluto/all/immagine.png)`) descrivendo ciò che *vedi* (le linee dell'errore di training vs validation). Spiega Test vs Validation, RMSE, MAE, R², citando i numeri e misurandoli sui grafici.
    *   **Capitolo 4: Anatomia dell'Errore e degli Outliers.** Estrapola i Top 10 Outliers da `Analysis_Log.txt`. Spiegane i significati atmosferici o caotici. Usa il grafico `Analysis_04_HourlyError.png` e descrivi esattamente la barra oraria più alta.
    *   **Capitolo 5: Lo Stress-Test e la Generalizzazione (Cross-Zone).** Estrapola i dati e cita `Predictions_Log.txt`. Inserisci le immagini delle zone esterne testate (es. TimeSeries Zone 8, 10, 11) commentando esattamente il comportamento della previsione rispetto al carico reale visualizzato.
    *   **Capitolo 6: Corredo Grafico e Diagnostica Visiva Dedotta.**
        DEVI INSERIRE FISICAMENTE IL COLLEGAMENTO ALLE IMMAGINI usando la sintassi Markdown `![Didascalia](../sessioni/2026_03_07/Report.../immagine.png)`. 
        Per *ogni immagine inserita* usata durante la tesi, descrivila rigorosamente in base a quello che *hai visto*:
        1. **Come lo hai dedotto tu come Intelligenza Analitica:** (es. *"Guardando la figura, noto che i pallini blu si discostano significativamente dalla diagonale a partire dai 300 kWh... "*).
        2. **La spiegazione "Junior":** Come un novellino deve leggere quel grafico (Asse X, Y, dispersione).

## Stile Vocale, Formattazione e Regole Auree

*   **Verbosità:** Dilata ogni singolo concetto. Sii estremamente esplicito; non porti limiti di spazio. Spiega le cose in modo prolisso.
*   **Citazione e Analisi Viva:** "Come leggiamo nella figura", "Osserviamo qui la linea a tratti". Devi comportarti come uno scienziato che sta indicando la slide durante un convegno.
*   **Immagini (CRITICO):** I percorsi DEVONO essere rigorosamente RELATIVI alla cartella `docs/` in cui salvi il file (es: `../sessioni/2026_03_07/Report_lstmExog.../immagine.png`). Non usare assolutamente percorsi assoluti, poiché i software di visualizzazione Markdown si rompono se il percorso assoluto contiene spazi o parentesi (es. `00) Tirocinio`).
*   **Terminologia.** Ad ogni espressione inglese (`Loss`, `Overfitting`, `Patience`, `Residuals`, `Scatter`) apri un inciso per definirne il ruolo fisico, etimologico e matematico partendo da zero.

## Prompt per Innescare
"Usa la skill lstm-report-junior-thesis-analyzer sulla cartella [INSERIRE PERCORSO CARTELLA REPORT]"
