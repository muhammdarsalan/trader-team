# Graph-Based Algorithmic Trading Research Platform

A modular, research-grade platform for developing, backtesting and paper-trading
systematic strategies, built around a graph/DAG decision architecture.

It exists to answer one question scientifically:

> **Which strategies work, under what market conditions, with what risk, and does
> combining them improve performance?**

---

## ⚠️ Read this first

- **This platform does not and will not execute real-money trades.** `LIVE` mode
  is rejected by the config validator. It is research, backtesting and paper
  trading only.
- **Nothing here predicts the market.** No component promises a profit, a win
  rate, or a monthly return. A strategy with a high win rate can still lose
  money; a strategy with a low win rate can be profitable. The platform measures
  expectancy and risk-adjusted performance, not win rate.
- **A good backtest is not evidence.** Out-of-sample results, walk-forward
  analysis and Monte Carlo distributions are, and even those are weak evidence.
  "This strategy does not work" is a successful research outcome.
- **Sophisticated architecture does not create profitability.** The graph is
  plumbing that lets independent components interact cleanly. The research
  process is the actual asset.
- Trading carries substantial risk of loss. Nothing here is financial advice.

---

## Status

**Phases 1–2 of 5 complete** — data, features, regimes, strategies.

| Phase | Scope | State |
|---|---|---|
| 1 | Config, data layer, quality engine, ingest | ✅ Complete |
| 2 | Features, regime detection, 5 strategies, signals | ✅ Complete |
| 3 | LangGraph workflow, selector, risk, execution, backtester | ⬜ Not started |
| 4 | Walk-forward, OOS, Monte Carlo, robustness, experiment DB | ⬜ Not started |
| 5 | Paper trading, Streamlit dashboard, docs | ⬜ Not started |

What works today: download and validate 26 years of daily OHLCV across six
instruments; compute a causal feature panel; classify market regimes; and run
five independent strategies that emit standardized, self-validating signals.

**No performance claims exist yet, by design.** Nothing has been backtested —
that is phase 3. Every strategy parameter is a hypothesis awaiting the
validation machinery of phase 4.

---

## Installation

Requires Python 3.11+ on Windows, macOS or Linux. No Docker, no GPU, no cloud,
no paid services.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Optional configuration:

```bash
copy .env.example .env        # Windows
# cp .env.example .env
```

The platform runs with **zero secrets**. `.env` only enables optional extras.

---

## Quick start

```bash
# Download the recommended XAUUSD research window
python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D --start 2012-01-01

# Whole configured universe, saving quality reports
python scripts/ingest_data.py --all --timeframe 1D --start 2012-01-01 --save-report

# Analyse a market: regime + what every strategy decided, and why
python scripts/analyze_market.py --symbol XAUUSD --timeframe 1D --history 250

# Run the tests
pytest -m "not network"
```

In code:

```python
from app.data.service import MarketDataService

svc = MarketDataService()
result = svc.get_historical_data("XAUUSD", "1D", start="2012-01-01")

print(result.quality.render())   # full data-quality report
df = result.df                   # validated canonical OHLCV
```

---

## Architecture

```
configs/*.yaml  ─→  app/config/     typed, validated settings
                         ↓
app/data/       provider → cache → cleaning → quality gate → resampling
                         ↓
app/features/   causal indicators, structure, volume
app/regimes/    rule-based market-state classification
app/signals/    the standardized Signal contract
app/strategies/ five independent strategies + registry
                         ↓
app/graph/      LangGraph DAG: state, nodes, routing       [phase 3]
app/risk/       position sizing and exposure limits        [phase 3]
app/execution/  realistic fill simulation                  [phase 3]
app/backtest/   event-driven engine and metrics            [phase 3]
app/ml/         optional classical ML                      [phase 4]
app/database/   SQLite experiment tracking                 [phase 4]
app/paper_trading/                                         [phase 5]
dashboard/      Streamlit UI                               [phase 5]
```

Design rules the codebase holds to:

- **Nothing is hard-coded.** Symbols, thresholds, risk limits and enabled
  strategies all live in `configs/*.yaml`, validated into pydantic models at
  startup so a typo fails loudly instead of silently mid-backtest.
- **One canonical data contract.** Every frame is tz-aware UTC, chronological,
  `float64` OHLCV. See [docs/data.md](docs/data.md).
- **Bars are labelled by open time.** At timestamp `t`, the bar's close is not
  yet known. This convention is what keeps look-ahead bias out.
- **Providers are swappable.** Strategy code never sees a vendor ticker.
- **Errors are isolated.** One failing component must not take down a run.
- **Features are causal, and it is enforced by test.** `assert_causal` recomputes
  every feature on a truncated series and compares; any difference means the
  full computation used future data. See [docs/features.md](docs/features.md).

---

## Configuration

| File | Purpose |
|---|---|
| `configs/platform.yaml` | Mode, kill switch, balance, base currency, seed |
| `configs/assets.yaml` | Instrument universe, tick sizes, spreads, vendor tickers |
| `configs/data.yaml` | Providers, cache, quality thresholds |
| `configs/features.yaml` | Indicator periods |
| `configs/regimes.yaml` | Regime classification thresholds |
| `configs/strategies.yaml` | Which strategies run, and their parameters |

Environment overrides: `TRADING_MODE`, `TRADING_ENABLED`, `GTP_LOG_LEVEL`,
`GTP_RANDOM_SEED`, `GTP_STARTING_BALANCE`, `GTP_BASE_CURRENCY`.

### The kill switch

`trading_enabled` defaults to **false** and ships that way. While false, no
order is placed in any mode — the graph still runs and still analyses, and every
blocked order is logged.

---

## Data

Free, no-login source: **Yahoo Finance chart API**. Six instruments configured
(XAUUSD, EURUSD, GBPUSD, USDJPY, US30, NASDAQ), timeframes 1M → 1D.

Two limitations that materially affect research, documented in full in
[docs/data.md](docs/data.md):

1. **XAUUSD is a proxy.** Yahoo publishes no spot gold series, so `GC=F` (COMEX
   front-month futures) stands in. Contract rolls create price steps that are
   not market moves.
2. **Use `--start 2012-01-01`.** Earlier Yahoo gold history is settlement-quality
   with impossible OHLC relationships, and the quality gate fails it on purpose.

Bring your own data any time: drop `data/raw/<SYMBOL>_<TIMEFRAME>.csv` and pass
`--provider csv`. Strategies and the backtester are unaffected.

---

## Testing

```bash
pytest -m "not network"    # default: fully offline and deterministic
pytest -m network          # the one live-endpoint test
pytest --cov=app           # with coverage
```

441 offline tests plus one live-endpoint test, covering schema contracts, config
validation, cleaning, resampling, the quality engine, cache integrity, all three
providers, the service facade, every indicator, market structure, the feature
engine, regime detection, the Signal contract and all five strategies.

Two families of test carry unusual weight:

- **Causality.** `assert_causal` recomputes a feature on a truncated series and
  compares the overlap; any difference proves future data was used. It covers
  every indicator, the feature engine end-to-end, the regime detector and the
  signals themselves.
- **Strategies must be able to fire.** A strategy that always returns `WAIT`
  satisfies every safety property while being useless, so each one is asserted
  to produce a signal under the conditions it targets.

---

## Security

- **Never commit secrets.** `.gitignore` excludes `.env`, `*.key`, `*.pem`,
  `kaggle.json` and `credentials.json`.
- Use `.env` for local values; `.env.example` documents them with no real values.
- The platform needs no credentials for research, backtesting or paper trading.
  Anything requiring a key is optional and off by default.
- Never commit broker credentials or API tokens. If one is ever exposed, rotate
  it — removing it from a later commit does not remove it from history.

---

## Project layout

```
app/            platform code (see Architecture)
configs/        YAML configuration
data/           raw / cache / external  (gitignored)
docs/           architecture, data, strategies, risk, backtesting
scripts/        CLI entry points
tests/          pytest suite
reports/        generated reports   (gitignored)
experiments/    experiment records  (gitignored)
```

---

## Documentation

- [docs/data.md](docs/data.md) — sources, provenance, schema, quality, limitations
- [docs/features.md](docs/features.md) — indicators, market structure, causality guarantees
- [docs/strategies.md](docs/strategies.md) — the five strategies, regimes, the Signal contract
- [docs/architecture.md](docs/architecture.md) — design decisions and rationale

---

## License and disclaimer

Research software provided as-is, with no warranty. Past performance does not
indicate future results. Backtested performance is hypothetical and carries
inherent limitations. Do not risk money you cannot afford to lose.
