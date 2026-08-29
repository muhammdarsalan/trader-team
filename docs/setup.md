# Setup and running

Every command on this page has been run against this commit. Where Windows and
Unix differ, both are given — the difference is always the virtual-environment
activation line and the file-copy command, never the Python invocation.

> **Nothing on this page can place a real trade.** There is no broker
> integration in this repository, `LIVE` mode is refused by the config
> validator, and the kill switch (`platform.trading_enabled`) ships `false`.
> See [What this will not do](#what-this-will-not-do).

---

## Requirements

- **Python 3.11 or newer.** 3.11 is what the project is tested on.
- About 400 MB of disk for the virtual environment and cached data.
- No Docker, no GPU, no cloud account, no paid API, no credentials.

Check your interpreter first — on Windows the `py` launcher is the reliable
way to pick a version:

```powershell
py -3.11 --version      # Windows
python3.11 --version    # macOS / Linux
```

---

## Install

### Windows (PowerShell)

```powershell
cd path\to\trader-team
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, it is the execution policy,
not the project. Allow signed local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Windows (Command Prompt)

```bat
cd path\to\trader-team
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd path/to/trader-team
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional `.env`

The platform runs with **zero secrets**. `.env` only enables optional extras
(alerting webhooks, an alternative data provider you would have to switch on
yourself). Copy it only if you want those:

```powershell
copy .env.example .env     # Windows
```
```bash
cp .env.example .env       # macOS / Linux
```

### Confirm the install

```bash
python -c "import app; from app.config.loader import get_config; print(get_config().platform)"
```

That line loads and validates all ten config files. If it prints a
`PlatformConfig`, the install is good.

---

## Running the tests

Once the virtual environment is active, the invocation is identical on every
platform:

```bash
pytest                                       # the default run: offline, deterministic
pytest -m "not slow and not network"         # skip the end-to-end validation studies
pytest -m network                            # the single live-endpoint test
pytest --cov=app                             # with coverage
pytest tests/test_paper_trading.py -q        # one file
```

The default run **excludes the one network test**. That deselection is
`addopts = "-q --strict-markers -m 'not network'"` in `pyproject.toml`, so a
clean checkout passes with no internet.

**A command-line `-m` replaces that expression, it does not extend it.** So
`pytest -m "not slow"` selects the live test back in and will fail offline —
spell out `-m "not slow and not network"` when you pass your own marker
expression. `pytest -m network` is the deliberate case: it runs only the live
test, and fails without a reachable Yahoo endpoint. That is the test doing its
job, not a defect.

Measured on this commit: the default run takes about **11 minutes** (935 tests),
most of it in the `slow`-marked validation studies. `-m "not slow and not
network"` takes about **4½ minutes** (906 tests).

---

## Running the platform

All five entry points are plain scripts that add the repository root to
`sys.path` themselves, so they work from any directory and need no
`pip install -e .` and no `PYTHONPATH`.

```bash
# 1. Download and validate history (writes a parquet cache under data/)
python scripts/ingest_data.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
python scripts/ingest_data.py --all --timeframe 1D --start 2012-01-01 --save-report

# 2. What the regime detector and every strategy say about a market right now
python scripts/analyze_market.py --symbol XAUUSD --timeframe 1D --history 250

# 3. Backtest, with a reproducible experiment record
python scripts/run_backtest.py --symbol XAUUSD --timeframe 1D --start 2012-01-01 --save

# 4. Validate a configuration out of sample, walk forward, under resampling
python scripts/run_research.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
python scripts/run_research.py --list-experiments      # what has already been tried

# 5. Paper trade
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --live
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --status

# 6. The monitoring dashboard
streamlit run dashboard/streamlit_app.py
```

`streamlit run` opens `http://localhost:8501` in your browser. It reads the
paper-trading state file written by step 5; with no session on disk it renders
and says so rather than showing a blank page or inventing numbers.

### Running without internet access

`ingest_data.py` and `run_paper_trading.py` take `--provider`.
`run_backtest.py` and `run_research.py` do **not** — they take the provider
from `configs/data.yaml`. To run those offline, point `GTP_CONFIG_DIR` at a
copy of `configs/` whose `default_provider` you have changed:

```powershell
xcopy /E /I configs %TEMP%\gtp-configs                       # Windows
# then edit %TEMP%\gtp-configs\data.yaml: default_provider: synthetic
$env:GTP_CONFIG_DIR = "$env:TEMP\gtp-configs"
python scripts\run_backtest.py --symbol XAUUSD --timeframe 1D --start 2015-01-01
```

```bash
cp -r configs /tmp/gtp-configs                               # macOS / Linux
# then edit /tmp/gtp-configs/data.yaml: default_provider: synthetic
GTP_CONFIG_DIR=/tmp/gtp-configs \
  python scripts/run_backtest.py --symbol XAUUSD --timeframe 1D --start 2015-01-01
```

`synthetic` generates a deterministic price series. It is a plumbing exercise
and a test fixture — **it is not a market, and no result computed on it says
anything about any strategy.** To work on real data offline instead, drop
`data/raw/<SYMBOL>_<TIMEFRAME>.csv` and use the `csv` provider.

### Environment overrides

| Variable | Effect |
|---|---|
| `TRADING_MODE` | `BACKTEST`, `PAPER`, `RESEARCH`, `ANALYSIS`. `LIVE` is refused. |
| `TRADING_ENABLED` | The kill switch. Must be `true` before any paper order is created. |
| `GTP_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `GTP_RANDOM_SEED` | Seed for every stochastic component |
| `GTP_STARTING_BALANCE`, `GTP_BASE_CURRENCY` | Portfolio starting point |
| `GTP_CONFIG_DIR` | Read `*.yaml` from somewhere other than `configs/` |
| `GTP_DATA_ROOT`, `GTP_REPORTS_DIR`, `GTP_EXPERIMENTS_DIR`, `GTP_PAPER_STATE_DIR`, `GTP_LOGS_DIR` | Relocate generated output |

Setting them:

```powershell
$env:TRADING_ENABLED = "true"     # PowerShell, current session only
set TRADING_ENABLED=true          # cmd.exe
```
```bash
export TRADING_ENABLED=true       # macOS / Linux
```

---

## Data limitations you need to know before reading any result

These are properties of the free vendor feed, not bugs, and they bound what
any number produced by this platform can mean. Measured figures and the full
discussion are in [docs/data.md](data.md).

**XAUUSD is a futures proxy, not spot gold.** Yahoo publishes no spot gold
series, so `GC=F` — COMEX front-month gold futures — stands in. Contract rolls
put price steps in the series that are not market moves, and futures carry a
basis that spot does not. Every surface that shows a price for XAUUSD labels it
a proxy: the persisted paper state, `--status`, the decision graph's market
node, and the dashboard's first banner. `is_proxy: true` in
`configs/assets.yaml` is what drives that, so the label cannot drift out of
sync with the data.

**Use `--start 2012-01-01` for gold.** Earlier Yahoo gold history is
settlement-quality and contains impossible OHLC relationships. The quality gate
fails it, deliberately. A backtest that silently began in 1999 would be
computing returns on prices that never traded.

**Yahoo's spot-FX high/low fields are unreliable.** In **2–6% of bars** across
EURUSD, GBPUSD and USDJPY, the reported high or low violates OHLC bounds — the
high is below the open or close, or the low is above it. Consequences:

- True range, ATR and every volatility measure derived from them are understated
  on affected bars.
- A backtested stop-loss may not trigger where it would have in the market.
- Any FX result carries this defect. It is not noise to ignore.

The quality engine counts these bars and reports the **share of the series**
they represent, because a raw count cannot be weighed against the thresholds in
`configs/data.yaml`. That share reaches the dashboard's data-quality table and
the risk engine's decision. A `FAIL` grade means the risk engine refuses to
size a position at all.

**A missing grade is treated as a refusal.** If no quality grade reaches the
risk engine it blocks with `DATA_QUALITY_UNKNOWN` rather than assuming a pass —
"nobody checked" must not size the same position as "checked and passed".

---

## What this will not do

- **It does not execute real-money trades, and cannot.** There is no broker
  client, no order-placement API, no venue credentials, and no code path from
  a decision to a real order. Fills come from `app/execution/simulator.py`,
  which computes them from historical or delayed bars.
- **`LIVE` mode is refused.** `TradingMode.LIVE` exists only so the config
  validator can reject it by name. Setting `TRADING_MODE=LIVE`, putting
  `mode: LIVE` in `platform.yaml`, or constructing `PlatformConfig(mode="LIVE")`
  all raise. So does building a config variant through `override_config`.
- **The kill switch ships off.** `platform.trading_enabled` is `false`, so no
  paper order is created until you turn it on. The graph still runs and still
  analyses; every blocked order is logged with its reason.
- **It does not predict anything.** No component forecasts a price, and no
  verdict in the research vocabulary means "profitable" — the strongest
  available conclusion is *survived this round of testing*.
- **A passing test suite says nothing about profitability.** The tests check
  that the code does what it claims. Whether a strategy has an edge is a
  separate question, answered — or more often not answered — by
  [docs/research.md](research.md).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'app'`** — the virtual environment is
not active, or you installed into a different interpreter. Re-activate and
re-check with `python -c "import app"`.

**`streamlit: command not found` / not recognised** — same cause. On Windows,
`.venv\Scripts\streamlit.exe` is the one you want; activating the environment
puts it on `PATH`.

**A `TypeError` about `width` or `use_container_width` from Streamlit** —
the pinned floor is `streamlit>=1.49`. Upgrade: `pip install -U streamlit`.

**Every Yahoo request fails with a connection or proxy error** — the endpoint
is unreachable from your network. The platform reports this as a fetch failure
and degrades honestly; it never substitutes invented data. Work from the cache,
a CSV drop, or `pytest -m "not network"`.

**The dashboard says `NO_DATA` / `IDLE`** — there is no paper-trading state
file yet. Run `scripts/run_paper_trading.py` once to create one. An empty
session reporting `IDLE` is correct behaviour, not a failure.
