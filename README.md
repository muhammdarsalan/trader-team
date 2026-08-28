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

**Phases 1–4 complete; phase 5 paper trading and dashboard complete.** Data,
features, strategies, graph, risk, execution, backtesting, the validation
machinery that tries to break it all, and a restartable paper-trading session
with a monitoring dashboard over it.

| Phase | Scope | State |
|---|---|---|
| 1 | Config, data layer, quality engine, ingest | ✅ Complete |
| 2 | Features, regime detection, 5 strategies, signals | ✅ Complete |
| 3 | LangGraph workflow, selector, risk, execution, backtester | ✅ Complete |
| 4 | Walk-forward, OOS, Monte Carlo, robustness, experiment DB | ✅ Complete |
| 5 | Paper trading engine, Streamlit dashboard, docs | ✅ Paper trading + dashboard complete |

What works today: download and validate 26 years of daily OHLCV across six
instruments; compute a causal feature panel; classify market regimes; run five
independent strategies through a LangGraph decision graph; weight them by
regime and measured performance; size positions against a full set of risk
limits; simulate fills with spread, slippage, commission and gaps; produce a
reproducible backtest with a complete trade and no-trade log; subject that
configuration to strict out-of-sample testing, walk-forward analysis,
Monte Carlo resampling of the trade sequence, parameter-neighbourhood sweeps and
data-snooping arithmetic, with every study recorded under a reproducible id; and
run a restartable paper-trading session — over replayed history or a refreshed
vendor feed — through the *same* execution model as the backtester, with a
dashboard that shows the regime, every strategy's vote, the risk verdict, the
decision graph and the realised result.

**The paper engine is not a second backtester.** It drives the same simulator,
risk engine, graph and portfolio through the same per-bar sequence, and a test
compares every field of every trade the two produce over one frame. A divergence
there would be invisible otherwise: both would keep producing plausible numbers.

Remaining for phase 5: research/paper integration, the final robustness and
presentation pass, and end-to-end release documentation.

**No profitability claim is made anywhere in this project.** The backtester
reports what happened on one historical sample, warnings included. On the
recommended XAUUSD window the current default configuration **loses money**
(−7.5% over 14 years, profit factor 0.96), and that number is printed exactly as
computed. The phase-4 validation report has no verdict meaning "profitable" and
never will; its best available conclusion is *survived this round of testing*,
and its most common one is that a configuration has not been shown to work.

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

# Run a backtest and save a reproducible experiment record
python scripts/run_backtest.py --symbol XAUUSD --timeframe 1D --start 2012-01-01 --save

# Validate a configuration out of sample, walk forward and under resampling
python scripts/run_research.py --symbol XAUUSD --timeframe 1D --start 2012-01-01

# What has already been tried against this data
python scripts/run_research.py --list-experiments

# Run the tests
pytest -m "not network"

# Paper-trade over replayed history (deterministic, no network once cached)
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --start 2012-01-01

# Fetch the newest bars and process whatever has not been seen
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --live

# Paper-trade your own file: drop data/raw/XAUUSD_1D.csv, then
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --provider csv

# Inspect a saved session without advancing it
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --status

# Launch the paper-trading dashboard
streamlit run dashboard/streamlit_app.py
```

Paper trading is simulation. The kill switch (`platform.trading_enabled`) ships
**false**, so no order is created until you turn it on, and even then no broker
is contacted — see [docs/paper_trading.md](docs/paper_trading.md).

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
app/graph/      LangGraph DAG: state, nodes, routing
app/risk/       position sizing and exposure limits
app/portfolio/  positions, equity curve, exposure
app/execution/  realistic fill simulation
app/backtest/   event-driven engine, metrics, provenance
                         ↓
app/research/   splits, walk-forward, Monte Carlo, robustness,
                overfitting diagnostics, SQLite experiment store
app/paper_trading/  restartable paper session, graph view, performance
dashboard/          Streamlit monitoring UI (shaping in view.py, rendering only
                    in streamlit_app.py, so the page is testable)
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
| `configs/risk.yaml` | Position sizing and every exposure limit |
| `configs/execution.yaml` | Spread, slippage, commission, fill assumptions |
| `configs/backtest.yaml` | Balance, warm-up, annualisation |
| `configs/research.yaml` | Splits, embargo, walk-forward, Monte Carlo, sweeps |

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
`--provider csv`. Strategies, the backtester and the paper runner are all
unaffected.

Both caveats travel with the data rather than living in one display, so every
surface that shows a price for a proxy series shows that it is a proxy — the
persisted paper state, the `--status` output, the decision graph's market node
and the first banner on the dashboard.

---

## Testing

```bash
pytest -m "not network"              # default: fully offline and deterministic
pytest -m "not network and not slow" # skip the end-to-end validation studies
pytest -m network                    # the one live-endpoint test
pytest --cov=app                     # with coverage
```

Offline tests covering schema contracts, config validation, cleaning,
resampling, the quality engine, cache integrity, all three providers, the
service facade, every indicator, market structure, the feature engine, regime
detection, the Signal contract, all five strategies, graph routing and error
isolation, signal aggregation and conflicts, position sizing and every risk
limit, order fills, slippage, spread, gaps, backtest causality, temporal splits,
the walk-forward selection rule, Monte Carlo resampling, parameter sweeps,
overfitting arithmetic, the experiment store, the paper-trading loop and its
persistence, and the dashboard — including a headless render of the actual
Streamlit page.

Eight families of test carry unusual weight:

- **Causality.** `assert_causal` recomputes a feature on a truncated series and
  compares the overlap; any difference proves future data was used. It covers
  every indicator, the feature engine end-to-end, the regime detector and the
  signals themselves.
- **Strategies must be able to fire.** A strategy that always returns `WAIT`
  satisfies every safety property while being useless, so each one is asserted
  to produce a signal under the conditions it targets.
- **The backtest is causal.** Running over the first N bars must produce exactly
  the trades a full run produced within that window. Any difference proves the
  full run consumed information that had not happened yet. The same test covers
  the equity curve.
- **Temporal separation is structural.** Evaluation windows must be ordered and
  disjoint, warm-up prefixes must never reach forward, and the embargo must
  actually separate. A contaminated split looks exactly like a good one in the
  output, so it is asserted mechanically instead of argued for.
- **Walk-forward selection cannot peek.** Each fold's choice must be
  reconstructible from that fold's training scores alone. Selection informed by
  the test window turns the whole analysis into an in-sample result and changes
  nothing visible about the report.
- **Paper trading and the backtester must agree.** Both are driven over one
  frame and every field of every shared trade is compared. They are two drivers
  of one execution model; if they diverged, both would keep producing plausible
  numbers and only this comparison would show they had stopped describing the
  same system.
- **A restart must change nothing.** A run split across a process boundary is
  compared, field by field, with one that never stopped — trades, cash, costs,
  the drawdown peak and the selector's measured performance. A restart bug never
  looks like an error: the drawdown peak resets and silently *widens* the risk
  limit, or excursions restart and every surviving trade reports that it never
  went against its entry.
- **Nothing may be fabricated.** An engine with no data must report `IDLE`, must
  colour no stage as having run, and must not present a starting balance as a
  measurement. Health was once reported `HEALTHY` whenever the error list
  happened to be empty, so an engine holding no data at all described itself as
  healthy while every stage of the graph showed green.

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
- [docs/graph_engine.md](docs/graph_engine.md) — the LangGraph workflow, state, routing, error isolation
- [docs/backtesting.md](docs/backtesting.md) — bar ordering, execution realism, risk limits, metrics
- [docs/research.md](docs/research.md) — validation, walk-forward, overfitting, experiment tracking
- [docs/paper_trading.md](docs/paper_trading.md) — the paper session, persistence, the dashboard, the graph view
- [docs/architecture.md](docs/architecture.md) — design decisions and rationale

---

## License and disclaimer

Research software provided as-is, with no warranty. Past performance does not
indicate future results. Backtested performance is hypothetical and carries
inherent limitations. Do not risk money you cannot afford to lose.
