---
name: stock-alert-monitor
description: Persistent US low-float momentum + catalyst screener that polls news feeds on an interval and pings on new matches — filters by price ($0.10–$10), float (<=20M), 30-day avg volume (>500K), relative volume (>2x), daily change (>+10%), and catalyst keywords (FDA, approval, contract, earnings, patent, partnership, acquisition, merge), then emits a VWAP-based BUY/WAIT/NO-ENTRY signal
---

# Stock Alert Monitor

Always-on watcher for catalyst-driven momentum in US low-float small/micro-cap
stocks. It polls news feeds, screens each catalyst headline's ticker against a
multi-factor filter, and pings you with a VWAP-based entry signal.

## When to Use This Skill

- You trade low-float momentum runners and need real-time catalyst + RVOL alerts
- You want a hands-off watcher that only fires when ALL screen criteria line up
- You want alerts pushed to chat, phone, or a Slack/Discord webhook

## Screen Criteria

An alert fires only when a catalyst headline's ticker passes **every** filter:

| Filter | Default | Flag |
|--------|---------|------|
| Share price | between **$0.10 and $10** | `--price-min` / `--price-max` |
| Float (low float) | **≤ 20,000,000** shares | `--max-float` |
| Avg daily volume (30-day) | **> 500,000** shares | `--min-avg-vol` |
| Relative volume (RVOL) | today vol / 30-day avg **> 2×** | `--min-rvol` |
| Daily % change | **> +10%** vs previous close | `--min-change-pct` |
| Catalyst keyword | FDA · Approval · Contract · Earnings · Patent · Partnership · Acquisition · Merge(r) | (in code) |

> RVOL rule for the agent: `Current Volume (Today) / Average Volume (30-day) > 2`.

If a data point can't be fetched (e.g. float is crumb-gated), that single filter
is **skipped** and the match is flagged `data_incomplete: true` — unless you pass
`--require-float`, which rejects tickers whose float can't be verified.

## Recommended Signal (VWAP-based)

Two modes — pick with `--signal-mode`:

**`momentum` (default — standard intraday logic):**
- price **below** VWAP → **NO ENTRY**
- price **≤ 3% above** VWAP (buy zone) → **BUY NOW**
- price **> 3% above** VWAP (extended) → **WAIT** pullback to VWAP / 9 EMA (1/3/5-min)

**`literal` (verbatim from the spec as written):**
- `|price − VWAP| / VWAP > 3%` → **BUY NOW**
- within 3% of VWAP → **WAIT** pullback to VWAP / 9 EMA
- below VWAP → **NO ENTRY**

The `vwap-dist-pct` threshold (default 3%) is configurable. VWAP and the 9 EMA are
computed from 1-minute intraday bars.

## Usage

```bash
# defaults: $0.10–$10, float<=20M, avg30>500K, RVOL>2, change>+10%, momentum signal
python skills/stock-alert-monitor/monitor.py --interval 120 --webhook "$ALERT_WEBHOOK"

# single pass (cron / one-shot)
python skills/stock-alert-monitor/monitor.py --once

# use the literal signal interpretation, require verified float
python skills/stock-alert-monitor/monitor.py --signal-mode literal --require-float
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--feeds URL ...` | M&A + biotech feeds | News sources to poll |
| `--interval N` | `300` | Seconds between polls |
| `--signal-mode` | `momentum` | `momentum` or `literal` VWAP logic |
| `--vwap-dist-pct P` | `3.0` | VWAP distance threshold (%) |
| `--require-float` | off | Reject tickers with unverifiable float |
| `--webhook URL` | `$ALERT_WEBHOOK` | POST each match to Slack/Discord |
| `--once` | off | Single pass then exit |
| `--state PATH` | `~/.cache/stock-alert-monitor/seen.txt` | De-dup memory |

## Running It Persistently

- **Inside Claude Code:** wrap the command in the `Monitor` tool (`persistent: true`)
  — each stdout JSON line becomes a chat/phone notification.
- **Standalone:** `cron`, `systemd`, or `nohup ... &`; use `--webhook` for pushes.
- **Via `/loop`:** in sandboxes where the shell has no egress, run a recurring agent
  turn that `WebSearch`es for fresh catalysts and `PushNotification`s on new matches.

## Output Schema

```json
{
  "ts": "2026-06-02T14:03:11Z",
  "ticker": "ABCD",
  "catalysts": ["fda", "approval"],
  "signal": "BUY NOW (+2.6% above VWAP, in buy zone)",
  "metrics": {"price": 4.20, "prev_close": 3.10, "change_pct": 35.5,
              "today_vol": 9100000, "avg_vol_30d": 1200000, "rvol": 7.6,
              "float": 8500000, "vwap": 4.09, "ema9": 4.01},
  "data_incomplete": false,
  "title": "ABCD Receives FDA Approval ...",
  "link": "https://..."
}
```

## Network Notes

Needs outbound access to your feed hosts and `query1.finance.yahoo.com` (price,
volume, daily bars). Float comes from the quoteSummary endpoint, which may be
rate-limited/crumb-gated — handled gracefully (filter skipped + flagged).

## Caveats

- A discovery/screening tool, not an execution feed; **verify every alert** and
  pair with the `risk-management` and `position-sizing` skills before trading.
- Quotes/intraday bars are delayed depending on the source; treat signals as
  directional guidance, not fills.
- Sub-$10 low-float names are highly volatile and prone to manipulation.
