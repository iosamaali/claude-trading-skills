#!/usr/bin/env python3
"""Persistent US small/micro-cap momentum + catalyst screener.

Polls one or more news feeds (RSS/Atom) on an interval, keyword-matches each
headline for catalysts, extracts the ticker, pulls intraday + daily market data,
applies a multi-factor screen, computes a VWAP-based entry signal, de-duplicates
against headlines already seen, and emits one line per NEW passing match on
stdout. It can also push each match to a Slack/Discord-style webhook.

Screen (all must pass unless data is unavailable — see notes):
  * Price                  : between $0.10 and $10.00
  * Float                  : <= 20,000,000 shares          (low float)
  * Avg daily volume (30d) : > 500,000 shares
  * Relative volume        : today_vol / avg30_vol > 2     (RVOL)
  * Daily % change         : > +10% vs previous close
  * Catalyst keyword       : FDA, Approval, Contract, Earnings, Patent,
                             Partnership, Acquisition, Merge(r)

Recommended signal (VWAP-based):
  * --signal-mode momentum (DEFAULT, standard intraday logic):
        price below VWAP                       -> NO ENTRY
        price <= 3% above VWAP (in buy zone)   -> BUY NOW
        price  > 3% above VWAP (extended)      -> WAIT pullback to VWAP / 9 EMA
  * --signal-mode literal (verbatim from the spec):
        |price - VWAP| / VWAP > 3%             -> BUY NOW
        price within 3% of VWAP                -> WAIT pullback to VWAP / 9 EMA
        price below VWAP                       -> NO ENTRY

Design notes
------------
* One JSON object per new passing match is printed to stdout (newline-delimited)
  so the script can be wrapped by Claude Code's `Monitor` tool (each line -> one
  notification). Works standalone too (cron, systemd, webhook).
* Market data comes from Yahoo's public chart API (no key). Float is best-effort
  via the quote/quoteSummary endpoint, which can be rate-limited or crumb-gated;
  when float is unavailable the float filter is SKIPPED (and the match is flagged
  data_incomplete=true) unless --require-float is set.
* Network egress to the feed hosts and query1.finance.yahoo.com is required.
  Some sandboxes allowlist outbound hosts; run where these are reachable.
* No third-party deps — stdlib only.

Usage
-----
    python monitor.py                          # defaults, 300s loop
    python monitor.py --interval 120 --once
    python monitor.py --signal-mode literal
    python monitor.py --require-float --webhook https://hooks.slack.com/...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from html import unescape
from pathlib import Path

# --- Catalyst keywords (lowercased substring match on headline + summary) ---
# "merge" intentionally also catches "merger" / "merges".
CATALYST_KEYWORDS = [
    "fda", "approval", "contract", "earnings",
    "patent", "partnership", "acquisition", "merge",
]

# --- Default screen thresholds (override via CLI) ---
DEFAULTS = dict(
    price_min=0.10,
    price_max=10.00,
    max_float=20_000_000,
    min_avg_vol=500_000,
    min_rvol=2.0,
    min_change_pct=10.0,   # percent, vs previous close
    vwap_dist_pct=3.0,     # percent threshold for the signal
)

DEFAULT_FEEDS = [
    "https://www.globenewswire.com/RssFeed/subjectcode/22-Mergers%20and%20Acquisitions/feedTitle/GlobeNewswire%20-%20Mergers%20and%20Acquisitions",
    "https://www.globenewswire.com/RssFeed/industry/4577-Biotechnology/feedTitle/GlobeNewswire%20-%20Biotechnology",
]

UA = "Mozilla/5.0 (stock-alert-monitor; +https://github.com/agiprolabs/claude-trading-skills)"

TICKER_RE = re.compile(r"\((?:NASDAQ|NYSE|NYSE American|AMEX|OTC)[:\s]+([A-Z]{1,5})\)|\$([A-Z]{1,5})\b")
ITEM_RE = re.compile(r"<(?:item|entry)\b.*?</(?:item|entry)>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


# ----------------------------- HTTP + feed parsing -----------------------------
def _http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _field(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", block, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return unescape(TAG_RE.sub("", m.group(1))).strip()


def parse_feed(xml: str) -> list[dict]:
    out = []
    for block in ITEM_RE.findall(xml):
        title = _field(block, "title")
        if not title:
            continue
        out.append({
            "title": title,
            "link": _field(block, "link"),
            "summary": _field(block, "description") or _field(block, "summary"),
        })
    return out


def matched_catalysts(text: str) -> list[str]:
    t = text.lower()
    return [k for k in CATALYST_KEYWORDS if k in t]


def extract_ticker(text: str) -> str | None:
    m = TICKER_RE.search(text)
    return (m.group(1) or m.group(2)) if m else None


# ----------------------------- Indicators -----------------------------
def vwap(highs, lows, closes, vols) -> float | None:
    num = den = 0.0
    for h, l, c, v in zip(highs, lows, closes, vols):
        if None in (h, l, c, v):
            continue
        tp = (h + l + c) / 3.0
        num += tp * v
        den += v
    return (num / den) if den else None


def ema(values, period: int) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < period:
        return None
    k = 2.0 / (period + 1)
    e = sum(vals[:period]) / period          # seed with SMA
    for v in vals[period:]:
        e = v * k + e * (1 - k)
    return e


# ----------------------------- Market data (Yahoo, best-effort) -----------------------------
def _chart(ticker: str, interval: str, rng: str, timeout: int) -> dict | None:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval={interval}&range={rng}")
    try:
        return json.loads(_http_get(url, timeout))["chart"]["result"][0]
    except Exception:
        return None


def fetch_metrics(ticker: str, timeout: int = 15) -> dict:
    """Return price/volume/VWAP/EMA9/float metrics. Missing fields -> None."""
    m: dict = {"ticker": ticker}

    # Intraday minute bars -> price, today volume, VWAP, EMA9, prev close
    intra = _chart(ticker, "1m", "1d", timeout)
    if intra:
        meta = intra.get("meta", {})
        q = intra.get("indicators", {}).get("quote", [{}])[0]
        closes = q.get("close") or []
        highs = q.get("high") or []
        lows = q.get("low") or []
        vols = q.get("volume") or []
        last_close = next((c for c in reversed(closes) if c is not None), None)
        m["price"] = meta.get("regularMarketPrice") or last_close
        m["prev_close"] = meta.get("chartPreviousClose") or meta.get("previousClose")
        m["today_vol"] = meta.get("regularMarketVolume") or sum(v for v in vols if v)
        m["vwap"] = vwap(highs, lows, closes, vols)
        m["ema9"] = ema([c for c in closes if c is not None], 9)

    # Daily bars (3mo) -> 30-day average volume
    daily = _chart(ticker, "1d", "3mo", timeout)
    if daily:
        dv = daily.get("indicators", {}).get("quote", [{}])[0].get("volume") or []
        dv = [v for v in dv if v]
        m["avg_vol_30d"] = (sum(dv[-30:]) / len(dv[-30:])) if dv else None
        if m.get("prev_close") is None:
            m["prev_close"] = daily.get("meta", {}).get("chartPreviousClose")

    # Derived: % change and relative volume
    if m.get("price") and m.get("prev_close"):
        m["change_pct"] = (m["price"] - m["prev_close"]) / m["prev_close"] * 100.0
    if m.get("today_vol") and m.get("avg_vol_30d"):
        m["rvol"] = m["today_vol"] / m["avg_vol_30d"]

    # Float (best-effort; often crumb-gated -> None)
    m["float"] = fetch_float(ticker, timeout)
    return m


def fetch_float(ticker: str, timeout: int = 10) -> int | None:
    url = ("https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
           f"{ticker}?modules=defaultKeyStatistics")
    try:
        data = json.loads(_http_get(url, timeout))
        ks = data["quoteSummary"]["result"][0]["defaultKeyStatistics"]
        return ks.get("floatShares", {}).get("raw")
    except Exception:
        return None


# ----------------------------- Screen + signal -----------------------------
def screen(m: dict, cfg) -> tuple[bool, list[str], bool]:
    """Return (passed, fail_reasons, data_incomplete)."""
    reasons: list[str] = []
    incomplete = False

    price = m.get("price")
    if price is None:
        return False, ["no price data"], True
    if not (cfg.price_min <= price <= cfg.price_max):
        reasons.append(f"price ${price:.2f} outside ${cfg.price_min}-${cfg.price_max}")

    chg = m.get("change_pct")
    if chg is None:
        incomplete = True
    elif chg <= cfg.min_change_pct:
        reasons.append(f"change {chg:+.1f}% <= +{cfg.min_change_pct}%")

    avg = m.get("avg_vol_30d")
    if avg is None:
        incomplete = True
    elif avg <= cfg.min_avg_vol:
        reasons.append(f"avg30 vol {avg:,.0f} <= {cfg.min_avg_vol:,}")

    rvol = m.get("rvol")
    if rvol is None:
        incomplete = True
    elif rvol <= cfg.min_rvol:
        reasons.append(f"RVOL {rvol:.2f} <= {cfg.min_rvol}")

    flt = m.get("float")
    if flt is None:
        incomplete = True
        if cfg.require_float:
            reasons.append("float unknown (--require-float)")
    elif flt > cfg.max_float:
        reasons.append(f"float {flt:,} > {cfg.max_float:,}")

    return (len(reasons) == 0), reasons, incomplete


def signal(m: dict, cfg) -> str:
    price, vw = m.get("price"), m.get("vwap")
    if price is None or vw is None or vw == 0:
        return "NO SIGNAL (no VWAP)"
    dist = (price - vw) / vw * 100.0          # signed % above/below VWAP
    thr = cfg.vwap_dist_pct
    if cfg.signal_mode == "literal":
        if price < vw:
            return "NO ENTRY (below VWAP)"
        if abs(dist) > thr:
            return f"BUY NOW ({dist:+.1f}% from VWAP)"
        return f"WAIT pullback to VWAP/9EMA ({dist:+.1f}% from VWAP)"
    # momentum (default)
    if dist < 0:
        return f"NO ENTRY (below VWAP {dist:+.1f}%)"
    if dist <= thr:
        return f"BUY NOW ({dist:+.1f}% above VWAP, in buy zone)"
    return f"WAIT pullback to VWAP/9EMA (extended {dist:+.1f}% above VWAP)"


# ----------------------------- Notify -----------------------------
def notify_webhook(url: str, text: str) -> None:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        print(f"# webhook error: {e}", file=sys.stderr, flush=True)


# ----------------------------- Main loop -----------------------------
def run_pass(cfg, seen: set) -> int:
    new_count = 0
    for feed in cfg.feeds:
        try:
            xml = _http_get(feed, timeout=cfg.timeout)
        except Exception as e:
            print(f"# feed error {feed}: {e}", file=sys.stderr, flush=True)
            continue
        for item in parse_feed(xml):
            key = item["link"] or item["title"]
            if key in seen:
                continue
            text = f"{item['title']} {item['summary']}"
            cats = matched_catalysts(text)
            if not cats:
                seen.add(key)
                continue
            ticker = extract_ticker(text)
            if not ticker:
                seen.add(key)
                continue
            seen.add(key)

            m = fetch_metrics(ticker, cfg.timeout)
            passed, reasons, incomplete = screen(m, cfg)
            if not passed:
                print(f"# skip {ticker}: {'; '.join(reasons)}", file=sys.stderr, flush=True)
                continue

            sig = signal(m, cfg)
            new_count += 1
            price = m.get("price")
            chg = m.get("change_pct")
            rvol = m.get("rvol")
            flt = m.get("float")
            headline = (
                f"🚀 {ticker} [{','.join(cats)}] ${price:.2f} "
                f"{chg:+.1f}% RVOL {rvol:.1f}x "
                f"float {(f'{flt/1e6:.1f}M' if flt else '?')} | {sig} :: {item['title']}"
            )
            match = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "ticker": ticker,
                "catalysts": cats,
                "signal": sig,
                "metrics": {k: m.get(k) for k in
                            ("price", "prev_close", "change_pct", "today_vol",
                             "avg_vol_30d", "rvol", "float", "vwap", "ema9")},
                "data_incomplete": incomplete,
                "title": item["title"],
                "link": item["link"],
                "text": headline,
            }
            print(json.dumps(match), flush=True)
            if cfg.webhook:
                notify_webhook(cfg.webhook, headline)
    return new_count


def load_seen(path: Path) -> set:
    return set(path.read_text().splitlines()) if path.exists() else set()


def save_seen(path: Path, seen: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(list(seen)[-5000:]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feeds", nargs="+", default=DEFAULT_FEEDS)
    p.add_argument("--interval", type=int, default=300, help="poll seconds")
    p.add_argument("--price-min", type=float, default=DEFAULTS["price_min"])
    p.add_argument("--price-max", type=float, default=DEFAULTS["price_max"])
    p.add_argument("--max-float", type=int, default=DEFAULTS["max_float"])
    p.add_argument("--min-avg-vol", type=int, default=DEFAULTS["min_avg_vol"])
    p.add_argument("--min-rvol", type=float, default=DEFAULTS["min_rvol"])
    p.add_argument("--min-change-pct", type=float, default=DEFAULTS["min_change_pct"])
    p.add_argument("--vwap-dist-pct", type=float, default=DEFAULTS["vwap_dist_pct"])
    p.add_argument("--signal-mode", choices=["momentum", "literal"], default="momentum")
    p.add_argument("--require-float", action="store_true",
                   help="reject tickers whose float can't be verified")
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument("--webhook", default=os.environ.get("ALERT_WEBHOOK"))
    p.add_argument("--state", default=os.path.expanduser(
        "~/.cache/stock-alert-monitor/seen.txt"))
    p.add_argument("--once", action="store_true", help="single pass then exit")
    cfg = p.parse_args()

    state = Path(cfg.state)
    seen = load_seen(state)
    print(f"# stock-alert-monitor up | feeds={len(cfg.feeds)} interval={cfg.interval}s "
          f"price=${cfg.price_min}-${cfg.price_max} float<={cfg.max_float:,} "
          f"avg30>{cfg.min_avg_vol:,} rvol>{cfg.min_rvol} chg>+{cfg.min_change_pct}% "
          f"signal={cfg.signal_mode} seen={len(seen)}", file=sys.stderr, flush=True)

    try:
        while True:
            n = run_pass(cfg, seen)
            save_seen(state, seen)
            if cfg.once:
                print(f"# pass complete, {n} new match(es)", file=sys.stderr, flush=True)
                return 0
            time.sleep(cfg.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
