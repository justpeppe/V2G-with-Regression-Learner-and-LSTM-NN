# Riepilogo Fix — LSTM Pipeline

## File modificati

### `Scripts/createLstmSequences.m`
**BUG #1 (CRITICO)** — Trasposizione sequenze:
```diff
- xTrainTemp{idxSequence} = xMat(i:tEnd, :);
+ xTrainTemp{idxSequence} = xMat(i:tEnd, :)';  % [numFeatures × numLags]
```
MATLAB's `lstmLayer` richiede input in formato **feature-major** `[C × T]`. Prima del fix, ogni sequenza era `[48 × 5]` (T×C) — la rete imparava le 5 feature come time steps e i 48 lag come feature, un errore devastante.

---

### `LSTM.m`

| Fix | Riga | Descrizione |
|-----|------|-------------|
| BUG #1 | 61 | `size(xTrain{1}, 1)` invece di `size(..., 2)` dopo trasposizione |
| BUG #3 | 69–79 | Architettura semplificata: 1 LSTM `OutputMode="last"` + 2 FC |
| BUG #4 | 39–57 | Context buffer di 48 righe per val/test, poi filtro sui giorni reali |
| BUG #6 | 88 | `LearnRateDropPeriod=50` (era 30) — decadimento LR meno aggressivo |
| BUG #8 | 94 | `ValidationPatience=15` — early stopping aggiunto |
| Stale ref | 134 | `validationWithCtx` invece di `validationNorm` nel `netStruct` |

**Architettura prima/dopo**:
```
PRIMA                              DOPO
lstmLayer(128, "sequence")  →      lstmLayer(128, "last")
dropoutLayer(0.2)                  dropoutLayer(0.2)
lstmLayer(64, "sequence")          fullyConnectedLayer(64)
dropoutLayer(0.2)                  reluLayer
lstmLayer(32, "last")              dropoutLayer(0.2)
fullyConnectedLayer(1)             fullyConnectedLayer(1)
```

---

## Verifica

- `checkcode(createLstmSequences.m)` → ✅ 0 warning
- `checkcode(LSTM.m)` → ✅ 0 warning
