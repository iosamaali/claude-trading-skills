#!/usr/bin/env python3
"""WOK signal classifier — predict >2% forward returns over the next 4 hours.

End-to-end pipeline built on the `feature-engineering`, `signal-classification`,
and `walk-forward-validation` skills:

  1. Load WOK 1h OHLCV + holder-count data (CSV, Birdeye+Helius live, or a
     clearly-labeled realistic synthetic fallback).
  2. Engineer the requested feature set:
       - RSI(14)
       - MACD (line, signal, histogram)
       - Bollinger Band width (20, 2σ)
       - Volume ratio (volume / rolling mean)
       - Holder-count momentum (multi-horizon % change)
  3. Label: 1 if the close 4 hours ahead is > +2% above the current close,
     else 0  (binary "does the next-4h move exceed +2%?").
  4. Walk-forward validation: 30-day train (720 bars), 7-day test (168 bars),
     1-day step (24 bars), 4-bar embargo gap (== forward horizon).
  5. Train an XGBoost classifier per window, collect out-of-sample predictions.
  6. Report classification metrics (per-fold + aggregate, confusion matrix)
     and feature importance (XGBoost gain + SHAP when available).

Usage:
    python wok_signal_classifier.py                 # synthetic fallback demo
    python wok_signal_classifier.py --csv wok_1h.csv
    python wok_signal_classifier.py --live          # Birdeye + Helius (needs keys)
    python wok_signal_classifier.py --days 120 --seed 7

CSV schema (--csv):
    timestamp, open, high, low, close, volume, holder_count
    (timestamp parseable by pandas; one row per 1h bar, ascending)

Dependencies:
    uv pip install pandas numpy scikit-learn xgboost
    uv pip install shap httpx   # optional (SHAP importance / live fetch)

Environment Variables (only for --live):
    BIRDEYE_API_KEY      Birdeye market-data key (OHLCV)
    HELIUS_API_KEY       Helius key (holder snapshots) — optional
    WOK_MINT             WOK token mint address on Solana

Disclaimer: research/educational tooling only. Not investment advice. Past
model performance does not guarantee future results.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ───────────────────────────────────────────────────
BARS_PER_DAY = 24                 # 1h bars

FORWARD_HORIZON = 4               # predict the move over the next 4 hours
RETURN_THRESHOLD = 0.02           # > +2% forward return == positive class

TRAIN_BARS = 30 * BARS_PER_DAY    # 30-day train window  = 720 bars
TEST_BARS = 7 * BARS_PER_DAY      # 7-day  test  window  = 168 bars
STEP_BARS = 1 * BARS_PER_DAY      # slide 1 day between windows = 24 bars
GAP_BARS = FORWARD_HORIZON        # embargo == forward horizon (no label leak)

# Indicator parameters
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD = 20, 2.0
VOLUME_WINDOW = 20
HOLDER_MOM_WINDOWS = [4, 12, 24]  # holder-count momentum horizons (hours)

FEATURE_COLUMNS = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "volume_ratio",
    "holder_mom_4",
    "holder_mom_12",
    "holder_mom_24",
]


# ════════════════════════════════════════════════════════════════════
# 1. DATA ACQUISITION
# ════════════════════════════════════════════════════════════════════
def generate_synthetic_wok(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic WOK 1h OHLCV + holder-count dataset.

    Memecoin-flavoured dynamics: regime switching, volatility clustering,
    volume spikes, and a holder base that grows on inflows. A *mild*, honest
    predictive relationship is embedded — rising holder momentum and volume
    surges modestly raise the odds of a >2% move over the next 4h — so the
    classifier has something real (but noisy) to learn, as it would on-chain.

    Args:
        n_days: Days of hourly data to generate.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame: timestamp, open, high, low, close, volume, holder_count.
    """
    rng = np.random.default_rng(seed)
    n = n_days * BARS_PER_DAY

    # Latent "demand" driver: persistent inflow pressure (AR(1)).
    demand = np.zeros(n)
    for i in range(1, n):
        demand[i] = 0.95 * demand[i - 1] + rng.normal(0, 1.0)

    # Holder count: grows with positive demand, sticky (rarely drops fast).
    holders = np.zeros(n)
    holders[0] = 5_000.0
    for i in range(1, n):
        growth = 0.0015 * max(demand[i], 0) + rng.normal(0, 0.0008)
        holders[i] = max(holders[i - 1] * (1.0 + growth), 100.0)

    # Returns: regime drift + a *leading* demand impulse + heteroskedastic
    # noise. Demand (which also drives holder growth) leads price by a few
    # hours, so holder momentum carries genuine forward information — the
    # realistic memecoin pattern the classifier is meant to detect.
    returns = np.zeros(n)
    vol = 0.03
    regime = 0
    for i in range(n):
        if rng.random() < 0.04:
            regime = rng.choice([-1, 0, 0, 1])
        vol = 0.85 * vol + 0.15 * abs(rng.normal(0, 0.018)) + 0.002
        lead = demand[i - 3] if i >= 3 else 0.0          # demand leads price
        demand_impulse = 0.011 * np.tanh(lead / 2.5)
        returns[i] = regime * 0.0015 + demand_impulse + rng.normal(0, vol)

    close = 0.0008 * np.exp(np.cumsum(returns))  # sub-cent memecoin price

    # OHLC around close.
    hi_wick = np.abs(rng.normal(0.012, 0.006, n))
    lo_wick = np.abs(rng.normal(0.012, 0.006, n))
    open_ = close * (1 + rng.normal(0, 0.006, n))
    high = np.maximum.reduce([close, open_]) * (1 + hi_wick)
    low = np.minimum.reduce([close, open_]) * (1 - lo_wick)

    # Volume: base + spikes that correlate with demand impulses.
    base_vol = 250_000 * np.exp(rng.normal(0, 0.4, n))
    spike = 1 + 3.0 * np.clip(demand / 4.0, 0, None) * (rng.random(n) < 0.25)
    volume = base_vol * spike

    timestamps = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "holder_count": np.round(holders).astype(int),
        }
    )


def load_csv(path: str) -> pd.DataFrame:
    """Load WOK 1h OHLCV + holder data from CSV.

    Args:
        path: CSV path with columns timestamp, open, high, low, close,
            volume, holder_count.

    Returns:
        Cleaned, time-sorted DataFrame.
    """
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume", "holder_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_live_wok(days: int) -> pd.DataFrame:
    """Fetch real WOK 1h OHLCV (Birdeye) and holder snapshots (Helius).

    Holder counts are typically only available as point-in-time snapshots, so
    this stitches the latest holder count across the window and forward-fills;
    for production you would persist periodic holder snapshots to a time series.

    Args:
        days: Lookback in days.

    Returns:
        DataFrame: timestamp, open, high, low, close, volume, holder_count.

    Raises:
        RuntimeError: If keys are missing or the API returns no data.
    """
    import httpx  # local import — only needed for --live

    api_key = os.getenv("BIRDEYE_API_KEY", "")
    mint = os.getenv("WOK_MINT", "")
    if not api_key or not mint:
        raise RuntimeError(
            "Live mode needs BIRDEYE_API_KEY and WOK_MINT (and optionally "
            "HELIUS_API_KEY) environment variables."
        )

    import time

    now = int(time.time())
    start = now - days * 86400
    headers = {"X-API-KEY": api_key, "x-chain": "solana", "accept": "application/json"}
    resp = httpx.get(
        "https://public-api.birdeye.so/defi/ohlcv",
        headers=headers,
        params={"address": mint, "type": "1H", "time_from": start, "time_to": now},
        timeout=30,
    )
    resp.raise_for_status()
    items = (resp.json().get("data") or {}).get("items") or []
    if not items:
        raise RuntimeError("Birdeye returned no OHLCV for WOK.")

    df = pd.DataFrame(items).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["timestamp"] = pd.to_datetime(df["unixTime"], unit="s", utc=True)

    # Holder count: best-effort single snapshot, forward-filled across window.
    holder_now = _fetch_holder_count_helius(mint)
    df["holder_count"] = holder_now if holder_now else np.nan
    df["holder_count"] = df["holder_count"].ffill().bfill()

    cols = ["timestamp", "open", "high", "low", "close", "volume", "holder_count"]
    return df[cols].sort_values("timestamp").reset_index(drop=True)


def _fetch_holder_count_helius(mint: str) -> Optional[int]:
    """Best-effort current holder count via Helius DAS, else None."""
    key = os.getenv("HELIUS_API_KEY", "")
    if not key:
        return None
    try:
        import httpx

        url = f"https://mainnet.helius-rpc.com/?api-key={key}"
        page, holders = 1, 0
        while True:
            r = httpx.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": "h",
                    "method": "getTokenAccounts",
                    "params": {"mint": mint, "page": page, "limit": 1000},
                },
                timeout=30,
            )
            owners = (r.json().get("result") or {}).get("token_accounts") or []
            if not owners:
                break
            holders += len({o.get("owner") for o in owners})
            page += 1
            if page > 50:  # safety cap
                break
        return holders or None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════
def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line, signal, macd_line - signal


def _bb_width(close: pd.Series) -> pd.Series:
    """Bollinger Band width = (upper - lower) / middle band."""
    mid = close.rolling(BB_PERIOD).mean()
    sd = close.rolling(BB_PERIOD).std()
    return (2 * BB_STD * sd) / mid


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the requested feature set and the >2% / 4h forward label.

    Normalizes MACD components by price so the model is scale-invariant
    across the dataset (MACD is in absolute price units otherwise).

    Args:
        df: OHLCV + holder_count frame (ascending by time).

    Returns:
        DataFrame with FEATURE_COLUMNS, `fwd_return`, and binary `label`.
    """
    out = df.copy()
    close = out["close"]

    out["rsi_14"] = _rsi(close)

    macd_line, macd_sig, macd_hist = _macd(close)
    # Scale MACD by price to keep it comparable across the series.
    out["macd_line"] = macd_line / close
    out["macd_signal"] = macd_sig / close
    out["macd_hist"] = macd_hist / close

    out["bb_width"] = _bb_width(close)

    vol_ma = out["volume"].rolling(VOLUME_WINDOW).mean()
    out["volume_ratio"] = out["volume"] / vol_ma

    for w in HOLDER_MOM_WINDOWS:
        out[f"holder_mom_{w}"] = out["holder_count"].pct_change(w)

    # Forward 4h return and binary label (>2% move == 1).
    out["fwd_return"] = close.shift(-FORWARD_HORIZON) / close - 1.0
    out["label"] = (out["fwd_return"] > RETURN_THRESHOLD).astype(float)
    out.loc[out["fwd_return"].isna(), "label"] = np.nan  # no future == no label

    return out


# ════════════════════════════════════════════════════════════════════
# 3. WALK-FORWARD VALIDATION
# ════════════════════════════════════════════════════════════════════
def walk_forward_splits(
    n_samples: int,
    train_size: int = TRAIN_BARS,
    test_size: int = TEST_BARS,
    step_size: int = STEP_BARS,
    gap: int = GAP_BARS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield time-ordered (train_idx, test_idx) with an embargo gap."""
    start = 0
    while start + train_size + gap + test_size <= n_samples:
        train_end = start + train_size
        test_start = train_end + gap
        train_idx = np.arange(start, train_end)
        test_idx = np.arange(test_start, test_start + test_size)
        yield train_idx, test_idx
        start += step_size


def make_model(scale_pos_weight: float):
    """Build an XGBoost classifier (sklearn GBM fallback if xgboost absent)."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        print("  (xgboost not installed — falling back to sklearn GBM)")
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, random_state=42,
        )


def run_walk_forward(feat: pd.DataFrame) -> dict:
    """Train per window and aggregate out-of-sample predictions.

    Args:
        feat: Output of build_features.

    Returns:
        Dict of per-fold metrics, aggregated OOS arrays, and mean importance.
    """
    X_all = feat[FEATURE_COLUMNS]
    y_all = feat["label"].to_numpy()
    ret_all = feat["fwd_return"].to_numpy()
    valid = ~np.isnan(y_all) & ~X_all.isna().any(axis=1).to_numpy()

    fold_metrics: list[dict] = []
    importances: list[np.ndarray] = []
    oos_pred, oos_true, oos_ret = [], [], []

    fold = 0
    for train_idx, test_idx in walk_forward_splits(len(feat)):
        tr = train_idx[valid[train_idx]]
        te = test_idx[valid[test_idx]]
        # Need both classes in train and enough test points.
        if len(tr) < 100 or len(te) < 20 or len(np.unique(y_all[tr])) < 2:
            continue
        fold += 1

        X_tr, y_tr = X_all.iloc[tr], y_all[tr]
        X_te, y_te = X_all.iloc[te], y_all[te]

        pos = max(int((y_tr == 1).sum()), 1)
        neg = int((y_tr == 0).sum())
        model = make_model(scale_pos_weight=neg / pos)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan
        except ValueError:
            auc = np.nan

        fold_metrics.append({
            "fold": fold,
            "n_train": len(tr),
            "n_test": len(te),
            "pos_rate": float((y_te == 1).mean()),
            "accuracy": accuracy_score(y_te, preds),
            "precision": precision_score(y_te, preds, zero_division=0),
            "recall": recall_score(y_te, preds, zero_division=0),
            "f1": f1_score(y_te, preds, zero_division=0),
            "auc": auc,
        })
        if hasattr(model, "feature_importances_"):
            importances.append(np.asarray(model.feature_importances_, dtype=float))

        oos_pred.extend(probs.tolist())
        oos_true.extend(y_te.tolist())
        oos_ret.extend(ret_all[te].tolist())

    if not fold_metrics:
        sys.exit("ERROR: no valid walk-forward folds — need more data.")

    return {
        "fold_metrics": fold_metrics,
        "oos_pred": np.array(oos_pred),
        "oos_true": np.array(oos_true),
        "oos_ret": np.array(oos_ret),
        "mean_importance": np.mean(importances, axis=0) if importances else None,
        "n_folds": fold,
    }


# ════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE (SHAP on a full-history model)
# ════════════════════════════════════════════════════════════════════
def shap_importance(feat: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Mean |SHAP| per feature from a model fit on all valid history."""
    try:
        import shap
    except ImportError:
        return None

    X = feat[FEATURE_COLUMNS]
    y = feat["label"].to_numpy()
    valid = ~np.isnan(y) & ~X.isna().any(axis=1).to_numpy()
    Xv, yv = X[valid], y[valid]
    if len(np.unique(yv)) < 2:
        return None

    pos = max(int((yv == 1).sum()), 1)
    model = make_model(scale_pos_weight=int((yv == 0).sum()) / pos)
    model.fit(Xv, yv)
    try:
        sv = shap.TreeExplainer(model).shap_values(Xv.iloc[-500:])
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════
# 5. REPORTING
# ════════════════════════════════════════════════════════════════════
def _bar(value: float, ref: float, width: int = 24) -> str:
    if ref <= 0:
        return ""
    return "#" * int(round(value / ref * width))


def print_report(results: dict, shap_df: Optional[pd.DataFrame]) -> None:
    fm = results["fold_metrics"]
    print("\n" + "=" * 72)
    print("WOK SIGNAL CLASSIFIER — WALK-FORWARD RESULTS")
    print(f"Target: close[t+{FORWARD_HORIZON}h] / close[t] - 1 > "
          f"{RETURN_THRESHOLD:.0%}   (1h bars)")
    print(f"Windows: train={TRAIN_BARS}b/30d  test={TEST_BARS}b/7d  "
          f"step={STEP_BARS}b/1d  gap={GAP_BARS}b")
    print("=" * 72)

    # Per-fold (head/tail to keep it readable).
    print(f"\n{'Fold':>4} {'Train':>6} {'Test':>5} {'Pos%':>6} {'Acc':>6} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 60)
    shown = fm if len(fm) <= 16 else fm[:8] + [None] + fm[-8:]
    for r in shown:
        if r is None:
            print(f"{'...':>4}")
            continue
        auc = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "  n/a"
        print(f"{r['fold']:>4} {r['n_train']:>6} {r['n_test']:>5} "
              f"{r['pos_rate']*100:>5.1f}% {r['accuracy']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {auc:>6}")

    # Aggregate out-of-sample.
    yp, yt = results["oos_pred"], results["oos_true"]
    pb = (yp >= 0.5).astype(int)
    print("-" * 60)
    print(f"\nAGGREGATE OUT-OF-SAMPLE  ({len(yt)} predictions, "
          f"{results['n_folds']} folds)")
    print("-" * 60)
    print(f"  Positive rate (actual >2% moves): {yt.mean()*100:5.1f}%")
    print(f"  Accuracy : {accuracy_score(yt, pb):.3f}")
    print(f"  Precision: {precision_score(yt, pb, zero_division=0):.3f}  "
          "(of predicted pumps, how many were real)")
    print(f"  Recall   : {recall_score(yt, pb, zero_division=0):.3f}  "
          "(of real pumps, how many we caught)")
    print(f"  F1       : {f1_score(yt, pb, zero_division=0):.3f}")
    if len(np.unique(yt)) > 1:
        print(f"  ROC-AUC  : {roc_auc_score(yt, yp):.3f}")
        print(f"  PR-AUC   : {average_precision_score(yt, yp):.3f}  "
              f"(baseline = {yt.mean():.3f})")

    tn, fp, fn, tp = confusion_matrix(yt, pb, labels=[0, 1]).ravel()
    print("\n  Confusion matrix @0.5      pred:0   pred:1")
    print(f"             actual 0    {tn:>7} {fp:>8}")
    print(f"             actual 1    {fn:>7} {tp:>8}")

    # Mean per-fold AUC stability.
    aucs = [r["auc"] for r in fm if not np.isnan(r["auc"])]
    if aucs:
        print(f"\n  Per-fold AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
              f"(stability across {len(aucs)} folds)")

    # XGBoost gain importance (mean across folds).
    imp = results["mean_importance"]
    if imp is not None:
        order = np.argsort(imp)[::-1]
        ref = imp[order[0]]
        print("\nFEATURE IMPORTANCE — XGBoost gain (mean across folds)")
        print("-" * 60)
        for rank, idx in enumerate(order, 1):
            print(f"  {rank:>2}. {FEATURE_COLUMNS[idx]:<16} "
                  f"{imp[idx]:.4f}  {_bar(imp[idx], ref)}")

    # SHAP importance.
    if shap_df is not None:
        ref = shap_df["mean_abs_shap"].max()
        print("\nFEATURE IMPORTANCE — mean |SHAP| (full-history model)")
        print("-" * 60)
        for rank, row in enumerate(shap_df.itertuples(index=False), 1):
            print(f"  {rank:>2}. {row.feature:<16} "
                  f"{row.mean_abs_shap:.4f}  {_bar(row.mean_abs_shap, ref)}")
    else:
        print("\n(SHAP not available — `uv pip install shap` for SHAP importance)")

    # Honest assessment.
    print("\nASSESSMENT")
    print("-" * 60)
    if len(np.unique(yt)) > 1:
        auc = roc_auc_score(yt, yp)
        if auc > 0.55:
            print(f"  ROC-AUC {auc:.3f} > 0.55 — model shows predictive edge.")
        else:
            print(f"  ROC-AUC {auc:.3f} ≤ 0.55 — weak/no edge; treat as noise.")
    print("  Note: '>2% in 4h' is rare and imbalanced — judge by PR-AUC and")
    print("  precision, not raw accuracy, and always net trading costs.")
    print("\n  Research/educational use only. Not investment advice.")
    print("=" * 72)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Path to WOK 1h OHLCV+holder CSV.")
    ap.add_argument("--live", action="store_true",
                    help="Fetch live data (needs BIRDEYE_API_KEY, WOK_MINT).")
    ap.add_argument("--days", type=int, default=120,
                    help="Days of synthetic data (default 120).")
    ap.add_argument("--seed", type=int, default=42, help="Synthetic RNG seed.")
    args = ap.parse_args()

    # 1. Data
    if args.csv:
        print(f"[1/4] Loading WOK data from {args.csv} ...")
        df = load_csv(args.csv)
        source = f"CSV ({args.csv})"
    elif args.live:
        print("[1/4] Fetching live WOK data (Birdeye + Helius) ...")
        df = fetch_live_wok(args.days)
        source = "LIVE (Birdeye/Helius)"
    else:
        print(f"[1/4] No --csv/--live — generating {args.days}d synthetic WOK data.")
        print("      (Set BIRDEYE_API_KEY + WOK_MINT and use --live for real data.)")
        df = generate_synthetic_wok(args.days, args.seed)
        source = f"SYNTHETIC ({args.days}d, seed={args.seed})"

    print(f"      Source: {source}")
    print(f"      Bars: {len(df)}  "
          f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

    # 2. Features + labels
    print("[2/4] Engineering features (RSI, MACD, BB width, volume ratio, "
          "holder momentum) ...")
    feat = build_features(df)
    valid = feat["label"].notna() & feat[FEATURE_COLUMNS].notna().all(axis=1)
    n_pos = int((feat.loc[valid, "label"] == 1).sum())
    n_neg = int((feat.loc[valid, "label"] == 0).sum())
    print(f"      Usable rows: {int(valid.sum())}  "
          f"(>2% moves: {n_pos}  |  others: {n_neg}  |  "
          f"base rate: {n_pos / max(n_pos + n_neg, 1) * 100:.1f}%)")

    # 3. Walk-forward
    print("[3/4] Running walk-forward validation (30d train / 7d test) ...")
    results = run_walk_forward(feat)
    print(f"      {results['n_folds']} folds, "
          f"{len(results['oos_pred'])} out-of-sample predictions.")

    # 4. SHAP
    print("[4/4] Computing SHAP feature importance ...")
    shap_df = shap_importance(feat)
    print("      done." if shap_df is not None else "      SHAP unavailable.")

    print_report(results, shap_df)


if __name__ == "__main__":
    main()
