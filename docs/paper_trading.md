# Paper Trading and the Monitoring Dashboard

How a simulated trading session runs, what survives a restart, and what the
dashboard will and will not tell you.

> **Nothing here executes a real trade.** `LIVE` mode is rejected by the config
> validator, no broker integration exists, and there is no flag that creates
> one. Every fill on this page comes from
> [`ExecutionSimulator`](../app/execution/simulator.py) applied to historical or
> delayed bars.

> **A paper result is not evidence.** It is one path through one sample, with
> the same standing as a backtest over the same bars — which is to say, the
> weakest kind. The dashboard reports what happened and never characterises it
> as a return, an edge, or a forecast.

---

## One execution model, two data sources

The paper engine is not a second backtester. It drives the *same* execution
model — the same [`ExecutionSimulator`](../app/execution/simulator.py), the same
[`RiskEngine`](../app/risk/engine.py), the same
[`TradingGraph`](../app/graph/workflow.py), the same
[`Portfolio`](../app/portfolio/portfolio.py) — through the same per-bar sequence
the [backtester](backtesting.md) uses:

```
For bar i:
  1. Fill the order queued at bar i-1, at bar i's OPEN
  2. Check exits for every open position against bar i's HIGH and LOW
  3. Track excursions, mark to market at bar i's CLOSE, snapshot equity
  4. Run the decision graph on data up to and including bar i
     → any order it creates is queued for bar i+1
```

`tests/test_paper_trading.py::test_paper_engine_reproduces_backtester_trades`
runs both engines over one frame and compares every field of every trade. That
test exists because a divergence here would be invisible: both engines would
keep producing plausible numbers, and only a bar-for-bar comparison shows they
had stopped describing the same system.

The two supported modes differ **only** in where the frame comes from.

### A. Historical deterministic replay

```bash
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
```

Bars come from `MarketDataService.get_historical_data`. Given the same inputs the
outcome is byte-identical, which is what makes a paper session comparable with a
backtest and with itself.

"Deterministic" here is asserted, not assumed. It regressed once for a reason
worth remembering: the synthetic provider seeded itself from `hash(symbol)`, and
Python randomises string hashing per process, so every run produced different
bars while the provider's docstring promised reproducibility. The defect was
invisible inside a single process — which is where it was always tested.
`test_replay_is_deterministic_across_processes` now digests the *data* as well as
the trades.

### B. Live-ish refresh

```bash
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --live
```

Bars come from `MarketDataService.get_latest_data`, which drops a still-forming
final bar — acting on a bar whose high, low and close are not yet determined is
look-ahead bias in live clothing. Whatever has not been seen is then processed
through the identical per-bar code. Nothing about this path is closer to a
broker than the replay path is.

`test_live_tick_uses_the_service_and_the_same_bar_logic` runs both paths over the
same bars and asserts an identical trade digest.

### Your own data

```bash
# drop data/raw/XAUUSD_1D.csv, then
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --provider csv
```

The free Yahoo feeds are caveated (see [docs/data.md](data.md)); bringing your
own bars is the documented way out, and the paper runner takes `--provider` like
every other entry point.

### Looking without touching

```bash
python scripts/run_paper_trading.py --symbol XAUUSD --timeframe 1D --status
```

Prints the saved session and exits. Processes nothing, writes nothing.

---

## What survives a restart

A paper session is long-lived, so the interesting bugs are the ones that only
appear across a process boundary. State is written atomically after every bar to
`data/paper/<SYMBOL>_<TIMEFRAME>.json` (temp file plus `replace`, so a crash
mid-write cannot leave truncated JSON) and reloaded on construction.

| Persisted | Why it cannot be recomputed |
|---|---|
| Cash, realised P&L, total costs | The trade history that produced them is bounded |
| Open positions | Including `entry_bar`, so `bars_held` continues correctly |
| Position excursions (MAE/MFE) | They depend on prices the position has already lived through |
| **Drawdown peak** | Not derivable from cash and positions — losing it silently *widens* the drawdown limit |
| **Daily-loss baseline** | Same: an account already 8% down would report no daily loss |
| Selector performance table | Losing it resets strategy weights to their priors, so behaviour would depend on when the process last restarted |
| Equity curve | So the dashboard chart survives a restart |

Two things are deliberately **not** taken from the file:

- **The kill switch.** `trading_enabled` is read from live configuration every
  time. A saved `true` must not re-enable order creation an operator has since
  switched off.
- **`live_trading_enabled`.** Recorded as `false` so it is auditable, never read.

A state file that will not parse is moved aside as `<name>.json.corrupt` and the
engine starts flat and says so. Continuing from a half-read file would mean
trading against positions that may or may not exist.

`test_restart_reproduces_continuous_run_exactly` splits a run across a restart
and asserts the whole outcome — every trade field, cash, costs, the drawdown
peak and the selector's table — matches a run that never stopped.

## Duplicate and replayed bars

A processed bar is skipped, not re-traded. The engine keeps a high-water mark
rather than a growing membership set, so a bar at or before the newest processed
timestamp is a no-op. This is what makes a restart that overlaps history safe,
and it is the live path's protection too: `get_latest_data` returns a window that
mostly repeats what the last tick already handled.

`process_bar(..., force=True)` re-runs a bar deliberately. Nothing does this
automatically.

A revised bar — same timestamp, different prices — is treated as a duplicate and
skipped. The alternative would be to unwind and re-run decisions that have
already been acted on, which is not something a live session can honestly do.

---

## Execution realism

Inherited wholesale from the [backtester's assumptions](backtesting.md), because
they are the same code:

- **Next-bar open fills.** A signal from bar `t`'s close fills at bar `t+1`'s
  open.
- **The spread is paid both ways.** Buys pay the ask, sells receive the bid, on
  entry *and* exit.
- **Slippage scales with volatility.** The `atr_fraction` model is handed the
  bar's real ATR. Omitting it would not disable slippage — it falls back to a
  flat 0.05%, silently replacing the configured model with a different one and
  making the paper run's costs differ from the backtest's.
- **Gaps are honoured.** A bar that opens beyond the stop fills at the open, not
  at the stop. Pretending otherwise deletes exactly the losses that hurt most.
- **An entry that gaps through its own stop is abandoned.** Such a position has
  a negative risk distance and would close on the bar that opened it.
- **Ambiguous bars resolve against you.** When one bar's range holds both the
  stop and the target, OHLC cannot say which came first.
  `execution.same_bar_resolution` decides (default: the stop), *and the fact
  that a resolution was needed is recorded* — on the trade, on the fill, in the
  event log, and as a count on the performance panel. A reader can see which
  trades rest on that assumption.
- **Risk is recomputed from the actual fill.** The fill moved with the spread
  and slippage, so the money genuinely at risk is not what the signal assumed.
  The stop level itself does not move: relocating it after seeing the fill would
  rewrite the trade in its own favour.

Queued orders are resolved on the next bar — filled, rejected or abandoned — and
then dropped. A market order does not wait around for a later bar, because no
real order would.

## Warm-up

No decision is taken inside the feature warm-up. Those bars are still
*processed*: equity is marked and any inherited position still has its stop
checked. What must not happen is a decision derived from indicators that are not
yet defined.

`backtest.warmup_bars` governs the paper run as well as the backtest. If they
resolved warm-up differently, an operator who lengthened it to stabilise a
backtest would find the paper run still deciding on bars the backtest had
discarded, and the two results would silently stop being comparable.

## Risk

Every limit in `configs/risk.yaml` applies unchanged: per-trade sizing, notional
caps, concurrent positions, portfolio risk, correlation, drawdown reduction and
halt, and the daily loss limit. The paper loop adds nothing and relaxes nothing.

The kill switch (`platform.trading_enabled`, **false** by default) blocks order
creation in every mode. With it on, the graph still runs, regimes are still
detected, strategies still signal and risk still evaluates — the block is
recorded as the decision, which is what makes the dashboard able to distinguish
"no signal" from "execution is switched off".

## Data quality

The [Phase 1 quality engine](data.md) is authoritative, and its verdict travels
with the data into risk:

| Grade | Effect |
|---|---|
| `PASS` | Sized normally |
| `WARNING` | Blocked when `risk.block_on_data_quality_warning` is set; otherwise traded, and the page says so in as many words |
| `FAIL` | Blocked. Every input to position sizing descends from that series |
| absent | Blocked. "Nobody checked" must not size the same position as "checked and passed" |

The gate already refuses to hand over `FAIL` data on the normal path, but that is
not the only path — a caller can disable validation or load a frame from disk.
Re-checking where money is committed is what makes it a gate rather than a
label.

The paper loop records the **whole report**, not just the grade. That matters
most for the FX feeds: Yahoo's OTC composites are a sampled quote series and
carry OHLC bound violations in a measurable share of bars. The engine's own
measurements are in [docs/data.md](data.md); the gate's thresholds in
`configs/data.yaml` decide whether that warns or fails, and the dashboard shows
the finding with its **share of the series**, because a 3%-of-bars defect and one
stale bar are not the same fact and the word "WARNING" cannot tell them apart.

## XAUUSD is a proxy

Yahoo publishes no spot gold series, so `GC=F` — COMEX front-month gold futures —
stands in. The asset carries `is_proxy: true` and a `data_caveat`, and the caveat
travels with the data rather than living in the dashboard, because every surface
that shows a price for a proxy has to show that it is a proxy and only one of
those surfaces is the dashboard. It appears in the persisted state, in the
`--status` output, on the market-data node of the graph, and as the **first**
banner on the page.

Contract roll, futures basis and exchange hours versus 24h spot all make this
series differ from spot XAUUSD in level and in where its gaps fall. Use
`--start 2012-01-01`: earlier Yahoo gold history is settlement-quality with
impossible OHLC relationships, and the quality gate fails it on purpose.

---

## The dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

It attaches to the same state file the runner writes — resolved through
`app.utils.paths.paper_state_path` by both, so the page cannot report `NO_DATA`
while the runner is happily trading into a different file. It creates no trades
and cannot reach a broker.

All shaping happens in [`dashboard/view.py`](../dashboard/view.py), which returns
plain data; [`dashboard/streamlit_app.py`](../dashboard/streamlit_app.py) only
renders it. So the numbers, the statuses and the graph layout are all assertable
without a browser — and `tests/test_dashboard_render.py` covers the remaining
risk that the page raises before drawing anything.

Every metric carries both a raw `value` and a formatted `display`. Tests read the
value, the page shows the display: a formatting change cannot quietly alter what
is asserted, and an assertion cannot pass on a correctly-formatted wrong number.

### What each tab answers

| Tab | Question |
|---|---|
| Market & regime | Is the data current and trustworthy, and what state does the system think the market is in? |
| Strategies & decision | What did each strategy say, what happened to its vote, and why was or was not a trade taken? |
| Portfolio | Where does the book stand — equity, realised and unrealised P&L, drawdown, open positions, recent trades? |
| Execution | What did the simulated fills cost, broken into spread, slippage and commission? |
| Performance | Of the closed trades, how did they turn out by strategy and by regime? |
| Graph | How far did this bar get through the pipeline, and what stopped it? |
| Activity | What has happened, and what has failed? |

### Performance is realised only

Every figure on the Performance tab comes from **closed trades**. An open
position has a mark, not an outcome, and letting one in means the table improves
whenever a losing trade is left running. Nothing is annualised, extrapolated or
projected.

Two honesty rules are built into the panel:

- **Thin samples are flagged.** Rows below the selector's own evidence threshold
  (`RegimePerformanceTracker.min_samples`, 10) are marked *sample only*. A mean R
  from three trades is noise, and presenting it beside one from ninety invites
  exactly the wrong reading. The selector will not let such a row move a weight;
  neither should a reader.
- **Profit factor with no losses is `None`, not infinity.** "Infinite profit
  factor" is an artefact of a short sample.

The aggregator signs its output `ensemble`, so a per-strategy table built on the
trade's `strategy` field would collapse to a single uninformative row. The paper
engine therefore carries the *contributing* strategy names through the order onto
the trade, and the panel breaks results down by contributor. A trade appears
under every strategy that voted for it, so those rows do **not** sum to the
portfolio's P&L — read each on its own: *of the trades this strategy argued for,
this is how they turned out.*

### Health is derived, not assumed

`system_health()` answers four independent questions — has anything run, did the
last refresh succeed, is the data current, and did anything fail — and reports
the **worst** finding rather than whichever check ran last.

| Status | Meaning |
|---|---|
| `IDLE` | No bar has been processed. Every figure is a starting value, not a measurement |
| `HEALTHY` | Bars processed, data current, quality graded, nothing failed |
| `DEGRADED` | A refresh failed, data is stale or ungradeable, quality is FAIL/UNKNOWN, or an error is recorded |

An earlier version reported `HEALTHY` whenever the error list happened to be
empty, which meant an engine holding no data at all described itself as healthy
while every stage of the graph showed green. `test_no_fabrication_when_nothing_has_run`
asserts the opposite of that behaviour, field by field.

Freshness is measured in **bar intervals**, not absolute time — four hours old
means something entirely different on a 1M feed than on a daily one — and
weekends are excluded for daily and coarser timeframes so a Monday check of a
Friday close is not reported as three intervals stale. Exchange holidays are not
modelled, and the freshness detail says so rather than smoothing it away.
`UNKNOWN` is not a synonym for fine: it means the age could not be established,
which for a monitoring surface is a problem in itself.

### The decision graph

```
Market Data → Features → Regime → Strategies → Selection
           → Aggregation → Risk → Execution → Paper Position
```

Each node carries two separate facts:

- **`status`** — what the stage concluded. `ACTIVE`, `OK`, `WAIT`, `WARMUP`,
  `SUPPRESSED`, `REJECTED`, `BLOCKED`, `ERROR`, `PENDING`, each with its own
  colour.
- **`reached`** — whether the stage actually ran.

Keeping them apart matters. A bar can complete every stage on data the quality
gate has caveated; the market node stays amber (that caveat is the reason risk
will refuse) while the path shows that the bar did in fact run all the way
through. Conflating the two once made a fully-completed decision report that
"nothing progressed past market data", which was simply false.

`path` is the trunk stages reached, in order. `stopped_at` and `stopped_reason`
name where the bar ended and why — a warm-up skip, an aggregate that declined, a
risk refusal, the kill switch. The view distinguishes:

- **active** nodes — on the path to a paper trade
- **suppressed** strategies — they signalled, and the selector weighted them to
  zero, so they could not contribute
- **rejected** strategies — they looked and declined
- **failed** nodes — the node errored, which is *not* the same as a strategy
  saying WAIT, and a page that rendered them identically would hide the failure
  it exists to catch
- **risk blocks** — with the block reason

Node coordinates are computed in
[`app/paper_trading/graph_view.py`](../app/paper_trading/graph_view.py) rather
than in the Streamlit layer, so the arrangement of the picture can be asserted
in a test without a rendering runtime.

### Where there is nothing to show

The page says so. It does not substitute a plausible-looking zero, and it does
not colour a stage green for something that never ran. An engine with no data
renders `NO_DATA`, `IDLE`, all-`PENDING`, an empty equity curve and a banner
saying nothing has run — and the sidebar tells you the command that would change
that.

---

## Limitations

- **No broker, and no path to one.** There is no order routing, no account
  reconciliation, no partial fills beyond the simulator's volume-participation
  rejection, and no margin model. `max_position_notional_pct` above 1.0 implies
  leverage the platform does not model.
- **A "live" tick is a delayed vendor feed, not a tick feed.** The freshest bar
  it can act on is the last *complete* one.
- **Costs are configured, not observed.** Spread comes from
  `assets.yaml:typical_spread`, not from a quote stream. On a real venue it
  widens exactly when it matters most.
- **Exchange holidays are not modelled** in the freshness check; one will show as
  an extra interval of delay.
- **The persisted trade log is bounded** (see `MAX_CLOSED_TRADES`). The
  selector's own sample table is not trimmed, so on a long session it can be
  weighting on trades that have aged out of the log — the Performance tab shows
  both tables and says which is which.
- **Synthetic data is a test fixture, not a market.** It has none of the fat
  tails, volatility clustering or autocorrelation of real prices. Never cite a
  result computed on it.
