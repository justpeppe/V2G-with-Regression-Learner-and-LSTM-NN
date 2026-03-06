# Guida all'Analisi per la Tesi Triennale - Rete LSTM

Questo documento riassume le analisi fondamentali da includere nella tesi e presentare alla commissione per dimostrare la solidità del modello di previsione energetica.

## 1. Strategia dei Predittori (Il Problema dello "Sfasamento")
Nelle serie storiche energetiche, la rete tende a diventare "pigra" e copiare semplicemente l'ultimo valore conosciuto (Modello di Persistenza).

*   **Problema:** Usando `AAC_energy(t-1)` per prevedere `AAC_energy(t)`, l'errore è minimo ma la previsione è traslata di uno step.
*   **Soluzione Proposta:** Usare solo feature esogene (meteo, orario) oppure una variabile **Lag-48** (consumo di ieri alla stessa ora).
*   **Difesa della tesi:** "Abbiamo rimosso il feedback immediato del consumo per costringere la rete a imparare i pattern reali del meteo e dell'orologio solare, evitando che il modello si limitasse a una banale ripetizione del dato precedente."

---

## 2. Metriche di Valutazione e Limitazioni
È fondamentale spiegare perché alcune metriche sono state preferite ad altre.

*   **RMSE (Root Mean Square Error):** Metrica principale. Da comparare con i colleghi nel valore assoluto (kWh).
*   **R² (Coefficiente di Determinazione):** Indica quanta varianza la rete riesce a spiegare (es. 0.85 = 85%).
*   **MAPE (Mean Absolute Percentage Error):** **DA ESCLUDERE o CITARE COME LIMITAZIONE.**
    *   *Perché:* Quando il consumo reale si avvicina a zero (notte), il calcolo del MAPE tende a infinito (`Inf`), rendendo la media priva di senso. In tesi, scrivi: "Il MAPE non è stato utilizzato come metrica principale a causa della presenza di valori prossimi allo zero nel dataset reale, che causano instabilità numerica."

---

## 3. Cosa controllare per la tesi (Tabella per la Commissione)

Questi sono gli elementi tipicamente attesi da una commissione per validare un modello di Deep Learning:

| Analisi | Strumento MATLAB | Perché serve |
| :--- | :--- | :--- |
| **Autocorrelazione del target** | `autocorr(y, NumLags=100)` | Giustifica la scelta di usare 48 campioni (24 ore) di "memoria" (numLags). |
| **Analisi dei Residui** | `plot(yTest - testPrediction)` | I residui (errore) devono essere casuali ("rumore bianco"). Se hanno un pattern, significa che la rete ha "saltato" qualche info importante. |
| **Scatter plot Reale vs Predetto** | `scatter(yTest, testPrediction)` + linea `y=x` | Mostra visivamente quanto le predizioni seguano la bisettrice. Più i punti sono sulla linea, più la rete è precisa. |
| **Tabella Comparativa** | — | Confronta RMSE e R² tuoi con quelli dei colleghi **sui medesimi giorni di test**. |
| **Curva di Training** | Grafico di `trainnet` | Dimostra che la rete ha imparato (errore scende) senza andare in overfitting eccessivo (linea validazione stabile). |

---

## 4. Analisi dell'Errore (Error Breakdown)
Prova a plottare l'errore medio per fasce orarie. Spesso le reti energetiche sbagliano di più nei momenti di "transizione" (alba/tramonto). Documentare *dove* la rete sbaglia è sinonimo di grande maturità scientifica in una tesi triennale.
