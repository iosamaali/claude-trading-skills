# WOK Signal Classifier

Predict whether **WOK** will move **> +2% over the next 4 hours** from 1h OHLCV
and on-chain holder data, using an **XGBoost** classifier validated with
**walk-forward** (30-day train / 7-day test) windows.

Built on the repo's `feature-engineering`, `signal-classification`, and
`walk-forward-validation` skills.

> ⚠️ Research/educational tooling only. **Not investment advice.** Past model
> performance does not guarantee future results.

---

## Feature set

| Feature | Definition |
|---|---|
| `rsi_14` | Wilder's RSI, 14 bars |
| `macd_line`, `macd_signal`, `macd_hist` | MACD(12, 26, 9), each normalized by price so the model is scale-invariant across the series |
| `bb_width` | Bollinger Band width = `(upper − lower) / middle`, 20-bar / 2σ |
| `volume_ratio` | `volume / rolling_mean(volume, 20)` |
| `holder_mom_4`, `holder_mom_12`, `holder_mom_24` | Holder-count momentum: % change in holder count over 4h / 12h / 24h |

**Label:** `1` if `close[t+4] / close[t] − 1 > 2%`, else `0`. Rows without a
full 4h future are dropped.

## Walk-forward validation

| Parameter | Value | Rationale |
|---|---|---|
| Train window | 720 bars (30 days) | Recent, enough to learn |
| Test window | 168 bars (7 days) | Statistically meaningful OOS block |
| Step | 24 bars (1 day) | Slide one day per fold |
| Embargo gap | 4 bars | Equals the forward horizon → no label leakage |

Each fold trains a fresh XGBoost model on the train window (with
`scale_pos_weight` for class imbalance), predicts the embargoed test window,
and out-of-sample predictions are aggregated across all folds.

## Usage

```bash
pip install -r requirements.txt

# Demo on a realistic, clearly-labeled synthetic WOK dataset
python wok_signal_classifier.py

# Your own data
python wok_signal_classifier.py --csv wok_1h.csv

# Live (Solana) — needs API keys
export BIRDEYE_API_KEY=...      # 1h OHLCV
export HELIUS_API_KEY=...       # holder snapshots (optional)
export WOK_MINT=<wok_mint>
python wok_signal_classifier.py --live
```

### CSV schema (`--csv`)

```
timestamp, open, high, low, close, volume, holder_count
```
One row per 1h bar, ascending by time; `timestamp` parseable by pandas.

## Data sources

This environment had no API keys and a network allowlist, so the demo runs on a
**synthetic** WOK series (memecoin-style regime switching, volatility
clustering, volume spikes, and a holder base that grows on inflows, with a
*mild, leading* holder→price relationship the model is meant to detect). The
code path for **real data** is implemented:

- **OHLCV** — Birdeye `GET /defi/ohlcv` (`birdeye-api` skill)
- **Holders** — Helius `getTokenAccounts` DAS (`token-holder-analysis` skill).
  Holder counts are usually point-in-time snapshots; for production, persist
  periodic snapshots into a holder time series rather than forward-filling.

## Outputs

- **Per-fold metrics** — accuracy, precision, recall, F1, ROC-AUC, positive rate
- **Aggregate OOS** — accuracy, precision, recall, F1, ROC-AUC, **PR-AUC vs
  base rate**, confusion matrix, and per-fold AUC stability (`mean ± std`)
- **Feature importance** — XGBoost gain (mean across folds) **and** mean |SHAP|
  from a full-history model

`sample_output.txt` contains a full demo run.

### Example (synthetic demo, seed 42)

```
AGGREGATE OUT-OF-SAMPLE  (13944 predictions, 83 folds)
  Positive rate (actual >2% moves):  33.9%
  Accuracy : 0.627
  Precision: 0.455
  Recall   : 0.505
  F1       : 0.478
  ROC-AUC  : 0.645
  PR-AUC   : 0.471   (baseline = 0.339)

FEATURE IMPORTANCE — mean |SHAP|
   1. holder_mom_4     ████████████████████████
   2. holder_mom_12    █████████
   3. holder_mom_24    ████████
   ...
```

> Because `>2% in 4h` is rare and imbalanced, judge the model by **PR-AUC and
> precision** rather than raw accuracy — and always evaluate signals **net of
> trading costs** (~50 bps round-trip). On real WOK data, expect a weaker edge
> than this synthetic demo.

## Notes & caveats

- **Holder data granularity** is the main real-world constraint — a true holder
  *time series* (not a single snapshot) is required for `holder_mom_*` to carry
  the forward information shown here.
- **Feature decay:** retrain frequently; monitor rolling per-fold AUC.
- **No look-ahead:** the embargo gap equals the 4h horizon; features use only
  past/current bars; labels use strictly future bars.
