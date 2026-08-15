# Data Layer

How market data enters the platform, what it is, and where it should not be trusted.

---

## 1. Source of record

**Provider:** Yahoo Finance public chart API
**Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/{ticker}`
**Cost:** free · **Authentication:** none · **Account:** none · **Retrieved:** 2026-08-15

No API key, no login, no rate-limit token. The platform calls the endpoint
directly with `requests` rather than through a wrapper library, so failures are
legible and the dependency surface stays small.

### Why not the originally-planned source

The first choice was Stooq, which historically served plain CSV over a URL. As
of this build Stooq gates that endpoint behind a JavaScript proof-of-work
browser check. Solving it would mean defeating bot detection, so the platform
does not do that and uses Yahoo instead.

Kaggle was excluded because it requires an account and an API token, which the
project must not depend on.

### Symbol mapping

| Platform symbol | Yahoo ticker | What it actually is |
|---|---|---|
| XAUUSD | `GC=F` | COMEX front-month gold futures (**proxy — see §2**) |
| EURUSD | `EURUSD=X` | Spot FX |
| GBPUSD | `GBPUSD=X` | Spot FX |
| USDJPY | `JPY=X` | Spot FX |
| US30 | `^DJI` | Dow Jones Industrial Average index |
| NASDAQ | `^IXIC` | Nasdaq Composite index |

Mapping lives in `configs/assets.yaml` under `provider_symbols`. Vendor tickers
never appear in strategy, risk or backtest code.

---

## 2. The gold proxy — read this before trusting a gold backtest

**Yahoo does not publish a spot XAUUSD series.** `XAUUSD=X`, `XAU=X` and
`GCUSD=X` all return HTTP 404. What the platform labels `XAUUSD` is therefore
`GC=F`, COMEX front-month gold futures.

Futures track spot gold closely but are not the same instrument:

- **Contract roll.** Yahoo's continuous series stitches successive front-month
  contracts. At each roll the price steps by the basis between contracts. That
  step is not a market move, but a breakout or momentum strategy cannot tell
  the difference and will happily trade it.
- **Basis / cost of carry.** Futures trade above spot by roughly the financing
  cost. The spread varies with interest rates, so the two series diverge slowly
  over long backtests.
- **Session hours.** COMEX has defined sessions and a daily settlement. Spot
  gold trades ~23h/day. Daily bars are therefore not the same 24h window, and
  intraday work on this series is materially different from intraday spot.
- **Volume is real** here, unlike spot — it is genuine exchange volume, which
  is why `has_reliable_volume: true` for XAUUSD while the FX pairs are `false`.

**To use true spot gold:** export a spot XAUUSD CSV from any broker
(MetaTrader, Dukascopy, HistData), drop it in `data/raw/XAUUSD_1D.csv`, set
`has_reliable_volume: false` for XAUUSD in `configs/assets.yaml`, and run with
`--provider csv`. Nothing else changes — not the strategies, not the
backtester.

---

## 3. Recommended research window

The quality gate was run across candidate start dates for `GC=F` daily bars:

| Start | Rows | Stale (O=H=L=C) | Invalid OHLC bounds | Zero volume | Gate |
|---|---|---|---|---|---|
| 2000-01-01 | 6,513 | 12.50% | 6.77% | 6.43% | **FAIL** |
| 2008-01-01 | 4,683 | 4.66% | 1.67% | 0.88% | **FAIL** |
| 2010-01-01 | 4,178 | 4.91% | 0.60% | 0.98% | **FAIL** |
| 2012-01-01 | 3,674 | 4.55% | 0.00% | 1.12% | WARNING |
| 2015-01-01 | 2,920 | 5.00% | 0.00% | 1.27% | WARNING |

Yahoo's early `GC=F` history is settlement-quality, not tick-derived: many bars
carry `open = high = low` with a differing close, and volume of 1 contract or
zero. Those are recorded settlement prints, not traded ranges. Any strategy
touching highs and lows — breakout, support/resistance, ATR stops, and every
stop-loss simulation — would be computing against fabricated ranges.

> **Use `--start 2012-01-01` for XAUUSD research.** That is 3,674 bars and ~14
> years, still comfortably over the ten-year requirement, and the raw data
> passes the OHLC-consistency check with zero repairs.

Pre-2012 data is not deleted and remains loadable; the quality gate simply
refuses to certify it, which is the intended behaviour rather than a bug.

---

## 4. Known defects in the current datasets

Measured on the 2012-01-01 → 2026-08-14 ingest:

| Symbol | Rows | Repaired bounds | Missing bars | Notes |
|---|---|---|---|---|
| XAUUSD | 3,674 | 0 | 140 (3.7%) | Missing = US market holidays |
| EURUSD | 3,805 | 102 (2.7%) | 0 | Yahoo FX high/low defects |
| GBPUSD | 3,806 | 78 (2.1%) | 0 | Yahoo FX high/low defects |
| USDJPY | 3,807 | 236 (6.2%) | 0 | Worst affected |
| US30 | 3,675 | 0 | 139 (3.6%) | 5 suspected bad prints |
| NASDAQ | 3,675 | 0 | 139 (3.6%) | — |

**Yahoo's spot-FX high/low fields are unreliable.** In 2–6% of bars the
recorded high sits below the open or close (or the low above them), which is
arithmetically impossible. Cleaning widens the wick to contain the body — the
minimal honest correction, since the true extreme is at least the open/close —
but the underlying high and low for those bars are simply wrong. Consequences:

- Range-derived features (ATR, Bollinger width, high/low breakouts) are
  understated on affected FX bars.
- Backtested stop-losses on FX may not trigger where they would have in
  reality, which flatters results.
- **This is not a reason to trust the FX results and move on.** For serious FX
  research, import broker CSVs via the `csv` provider.

The "missing bars" counts for XAUUSD/US30/NASDAQ are public holidays
(2012-01-16 = MLK Day, 2012-02-20 = Presidents' Day, 2012-04-06 = Good Friday,
and so on). The quality engine deliberately counts them rather than pretending
to know every venue's holiday calendar, and labels the finding accordingly.

---

## 5. Pipeline

```
Provider (yahoo / csv / synthetic)
        ↓ canonical schema coercion
Cache (parquet + manifest with SHA-256 checksum)
        ↓
Cleaning  → CleaningReport
        ↓
Resampling (only ever coarser)
        ↓
Quality gate → DataQualityReport (PASS / WARNING / FAIL)
        ↓
Strategies, features, backtester
```

Entry point for everything downstream:

```python
from app.data.service import MarketDataService

svc = MarketDataService()
result = svc.get_historical_data("XAUUSD", "1D", start="2012-01-01")

result.df        # canonical OHLCV DataFrame
result.quality   # DataQualityReport
result.cleaning  # CleaningReport
```

---

## 6. The canonical schema

Every frame that crosses a module boundary satisfies this contract, and
downstream code is entitled to assume it without re-checking:

| Property | Guarantee |
|---|---|
| Index | tz-aware **UTC** `DatetimeIndex`, named `timestamp` |
| Ordering | strictly increasing, no duplicates |
| Columns | `open, high, low, close, volume`, all `float64` |
| OHLC | no NaN; all > 0; `high ≥ max(open, close)`; `low ≤ min(open, close)` |
| Volume | may be NaN where the venue publishes none |

### Bars are labelled by their OPEN time

A 4H bar stamped `12:00` covers `[12:00, 16:00)`. At timestamp 12:00 **the
close is not yet known.**

This is the single most important convention in the platform. Resampling uses
left-closed, left-labelled windows for exactly this reason. A right-labelled
bar invites a strategy to read a 16:00-stamped bar "at 16:00" and act on a
close it could not have known — look-ahead bias that produces beautiful,
worthless backtests.

---

## 7. Timeframes

Supported: `1M, 5M, 15M, 30M, 1H, 4H, 1D`.

Yahoo serves `1m, 5m, 15m, 30m, 1h, 1d` natively. **4H is built by resampling
1H**, automatically, by `MarketDataService`.

**Resampling only ever goes coarser.** Requesting 1H from a 1D source raises
`ResampleError` rather than interpolating: the intrabar price path is
unknowable, and reconstructing it fabricates prices that never traded.

### Yahoo intraday retention limits

| Interval | History available |
|---|---|
| 1m | ~7 days |
| 5m–1h | ~730 days |
| 1d | full history |

Requests beyond these raise `DataUnavailableError` rather than returning a
silently truncated series. **Long intraday backtests are not possible on this
provider** — use the `csv` provider with broker exports.

---

## 8. The quality gate

`DataQualityEngine` grades every series **PASS / WARNING / FAIL**. A FAIL raises
`DataQualityError` and the data never reaches a strategy (configurable via
`fail_on_quality_fail`).

Checks: row count · timezone/ordering · duplicates · NaN prices · non-positive
prices · OHLC consistency · missing bars · large gaps · price jumps · stale
bars · volume sanity · freshness · **cleaning repairs**.

Two checks are worth understanding:

**Price jumps use median absolute deviation, not standard deviation.** A single
bad print inflates the standard deviation enough to hide itself. The median
does not move. Detection is at 12 robust sigma by default.

**Cleaning repairs are reported to the gate.** Cleaning runs before validation,
so a repaired series looks pristine by the time it is graded. The
`CleaningReport` is passed into `validate()` so the report states how much
surgery the vendor data needed — this is what surfaced the FX high/low problem
in §4, which was otherwise completely invisible.

### Missing bars vs session breaks

Counting missing bars exactly needs a per-venue session calendar, which no free
source provides reliably. Rather than pretend, the engine separates:

- **Missing bars** — gaps of 1.5–3.5 bar intervals: genuinely dropped data.
- **Large gaps** — anything bigger: weekends, holidays, closures. Reported for
  visibility, never fatal.

Daily series instead use exact business-day arithmetic, where holidays remain a
known and documented source of over-counting.

---

## 9. Cleaning policy

| Defect | Action |
|---|---|
| Out of order | Sort |
| Duplicate timestamp | Keep last (configurable) |
| NaN in OHLC | **Drop the bar** |
| Zero/negative price | **Drop the bar** |
| `high < low` | **Drop the bar** — unrepairable |
| `high < max(open, close)` | Widen the wick to contain the body |
| `low > min(open, close)` | Widen the wick to contain the body |
| Negative volume | Set to NaN (unknown, not zero) |

**Cleaning never interpolates a price.** An interpolated bar is a price at
which nobody could have traded. Unrepairable bars are dropped and counted, and
every action appears in the `CleaningReport`.

Order matters: `high < low` is handled *before* bound repair, because clamping
both ends to the body would "fix" a corrupt bar into a zero-range one.

---

## 10. Cache and provenance

Location: `data/cache/{provider}/{SYMBOL}_{TIMEFRAME}.parquet`
Manifest: `..._{TIMEFRAME}.manifest.json`

Each manifest records symbol, timeframe, provider, row count, date range,
SHA-256 content checksum, write time, and the provider's own metadata. On load
the checksum is re-verified; a mismatch is treated as a cache miss rather than
silently trusted.

Writes go to a temp file and are atomically swapped, so an interrupted run
cannot leave a half-written payload that a later run would believe.

Cached payloads are gitignored. Manifests are small and may be committed if you
want dataset versions pinned in git history.

---

## 11. Using your own CSV files

Drop `data/raw/<SYMBOL>_<TIMEFRAME>.csv`, e.g. `XAUUSD_1D.csv`, then:

```bash
python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D --provider csv
```

The reader auto-detects separator (`, ; \t |`), decimal mark (handles European
`;`-separated, `,`-decimal exports), encoding and timestamp format, and accepts
common vendor column spellings (`Date`, `Time`, `O/H/L/C`, `Vol`, `TickVol`, …).

Naive timestamps are assumed **UTC**. If your export is in broker or venue
local time, set it explicitly — guessing would silently shift every bar:

```yaml
providers:
  csv:
    options:
      assume_timezone: "America/New_York"
```

---

## 12. Synthetic data

`SyntheticProvider` generates deterministic geometric Brownian motion so the
whole pipeline is testable offline.

**It is a test fixture, not a research tool.** It has none of the fat tails,
volatility clustering or autocorrelation of a real market. Every series it
produces is tagged `metadata["synthetic"] = True`. Never cite a backtest
computed on it as evidence a strategy works. Its legitimate uses: exercising
code paths, edge cases, and confirming a strategy is *not* profitable on noise.

---

## 13. Reproducing the datasets

```bash
# Recommended research window
python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D --start 2012-01-01

# Whole universe, with quality reports written to reports/data_quality/
python scripts/ingest_data.py --all --timeframe 1D --start 2012-01-01 --save-report

# Force re-download
python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D --refresh
```

Data is regenerable from these commands, which is why `data/` is gitignored.
