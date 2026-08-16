# Architecture

Design decisions and the reasoning behind them. Updated as each phase lands.

---

## Guiding principles

Ordered. When two conflict, the higher one wins.

1. **Correctness > sophistication.** A simple component that is right beats a
   clever one that is subtly wrong. In this domain "subtly wrong" means a
   backtest that looks profitable and is not.
2. **Risk management > return chasing.**
3. **Out-of-sample evidence > backtest beauty.**
4. **Reproducibility > complexity.** A result nobody can regenerate is not a
   result.
5. **Free/open-source > paid services.**

---

## Layering

Each layer depends only on those above it. Nothing reaches sideways or upward.

```
app/utils/       paths, logging, time      — depends on nothing
app/config/      typed settings            — depends on utils
app/data/        market data pipeline      — depends on config, utils
app/features/    indicators                — depends on data          [phase 2]
app/regimes/     market-state detection    — depends on features      [phase 2]
app/strategies/  signal generation         — depends on features, regimes [phase 2]
app/graph/       orchestration             — depends on all of the above  [phase 3]
```

A strategy cannot import the backtester. The backtester cannot import a specific
strategy. Both meet at the interfaces in `app/strategies/base.py` and the graph
state, which is what makes strategies addable without touching anything else.

---

## Decisions

### Bars are labelled by open time

A 4H bar stamped `12:00` covers `[12:00, 16:00)`. At `12:00` its close is
unknown.

The alternative — labelling by close — reads more naturally and quietly invites
look-ahead bias: a strategy handed a bar "at 16:00" acts on a close it could not
have known at any point during that window. Every resample in the platform is
left-closed and left-labelled for this reason, and the backtester will act on
the *next* bar's open.

This is the single convention most likely to be broken by a future
contributor, so it is stated in `app/data/schema.py`, `resample.py`,
`docs/data.md` and here.

### One canonical data contract

Every OHLCV frame crossing a module boundary is tz-aware UTC, chronological,
unique-indexed, `float64`, with valid OHLC relationships. Downstream code
asserts this once at the boundary and then trusts it.

The alternative — every consumer defensively re-checking — produces either
duplicated validation everywhere or, more commonly, none at all.

### Cleaning reports to the quality gate

Cleaning runs *before* validation, which means a repaired series looks pristine
by the time it is graded. The `CleaningReport` is therefore passed into
`DataQualityEngine.validate()`, and heavy repair raises a finding.

This was not theoretical. It immediately surfaced that Yahoo's spot-FX feed has
impossible high/low values in 2–6% of bars (USDJPY worst at 6.2%) — a defect
that was completely invisible while cleaning silently fixed it.

### Cleaning never interpolates

Bars that cannot be repaired are dropped, never filled. An interpolated bar is a
price at which nobody could have traded, and a backtest that fills on one is
measuring fiction.

Widening a wick to contain the candle body *is* allowed: when `high < close`,
the true high was at least the close, so the correction is minimal and the
direction is known. Inventing a whole bar is not the same operation.

### Resampling refuses to upsample

1H → 4H is aggregation. 4H → 1H is fabrication: the intrabar path is
unrecoverable, and any reconstruction systematically flatters strategies that
depend on intrabar extremes (every stop-loss simulation). `ResampleError` is
raised rather than guessing.

### Providers behind an interface

`MarketDataProvider` has two methods. Yahoo, CSV and synthetic implement it;
strategies never see a vendor ticker, and swapping sources changes one config
line. Symbol mapping lives in `configs/assets.yaml`.

### Outliers via median absolute deviation

Bad-print detection uses MAD, not standard deviation. A single extreme print
inflates the standard deviation enough to hide itself — the statistic is
corrupted by the very observation it should flag. The median does not move.

### Missing bars vs session breaks

Counting missing bars exactly requires a per-venue session calendar that no free
source provides. Rather than pretend precision it lacks, the engine reports two
separate things: small gaps (dropped bars) and large gaps (weekends, holidays,
closures). Daily series use exact business-day arithmetic, with holiday
over-counting stated openly in the report text.

Being visibly approximate beats being invisibly wrong.

### Cache integrity is checked, not assumed

Every cached payload carries a SHA-256 content checksum in its manifest,
re-verified on load; a mismatch is a cache miss. Writes go through a temp file
and an atomic swap, so an interrupted run cannot leave a truncated payload that
a later run believes.

Forcing this to actually work exposed that pandas' default CSV reader is lossy
(~1e-14), which required `float_precision="round_trip"` on read and `%.17g` on
write. A checksum that cries wolf gets ignored, which is worse than no checksum.

### Configuration is validated, and unknown keys are errors

All config goes through pydantic models with `extra="forbid"`. A misspelled key
is a startup error, not a silently ignored setting that produces mysteriously
wrong results three weeks later.

### Features publish confirmed structure, not detected structure

A swing pivot is defined by bars on both sides of it, so it is not knowable at
the bar where it occurs. Publishing it there is the most common source of
look-ahead bias in a trading platform, and it produces backtests that look
excellent and fail live.

`swing_points()` publishes a pivot `right` bars later, at the moment
confirmation actually arrives, while preserving the pivot's true price and age.

### Causality is enforced by test, not by review

`tests/helpers.py::assert_causal` computes a feature on the first *n* bars,
computes it again on the full series, and compares the overlap. Any difference
proves the full computation consumed future data.

Reading code cannot establish this reliably; a shifted window or a
`rolling(center=True)` is easy to miss and catastrophic. Every indicator, every
structure feature, the whole feature engine, the regime detector and all five
strategies are covered.

### A breakout is a transition, not a state

`breakout_up` is true whenever price is beyond the channel — which, in a
sustained trend, is every bar for hundreds of bars. Consumers deciding "is this
a breakout right now" use `breakout_up_fresh`, which additionally requires the
channel to have been intact just before.

This was found by a test asserting a clean linear uptrend classifies as
`TRENDING_UP`; without freshness the detector reported `BREAKOUT` permanently
and never once identified the trend.

### Strategies decline rather than raise

`generate_signal()` returns `WAIT` with a recorded reason when preconditions
fail — short history, unsupported timeframe, warm-up, missing indicators. One
strategy being misconfigured must not interrupt the other four, and phase 3's
graph will mark a genuinely failing strategy `UNAVAILABLE` and continue.

### Confidence is not a probability

A strategy's confidence measures how well *its own conditions* were met. It is
not calibrated against outcomes and nothing treats it as a win probability.
Conflating the two would make the ensemble weighting in phase 3 meaningless.

### The graph creates orders; it never fills them

A fill requires the *next* bar's prices, and the graph only ever sees the bar it
is analysing. Letting it fill would put look-ahead bias into the architecture
itself, where no later test could remove it. The backtester and paper loop own
the fill, because only they can advance time.

### Execution assumptions default to pessimistic

Buys pay the ask, sells receive the bid, slippage scales with ATR, gaps fill at
the open, and a bar containing both stop and target resolves as a **loss**.

That last one is the sharpest: OHLC data genuinely cannot say which level was
touched first, and assuming the target means believing the flattering
possibility every single time. `target_first` exists in the config so the
sensitivity can be measured, not so it can be used.

### Cleaning-style honesty applied to costs

Entry costs *and* exit costs are charged. Ignoring exit costs roughly halves the
modelled cost of a round trip, which is the difference between a marginal
strategy and an apparently profitable one — as the current XAUUSD backtest
demonstrates, where costs exceed gross trading profit.

### Correlated positions share one risk budget

Three longs in EURUSD, GBPUSD and AUDUSD are one short-dollar bet wearing three
hats. Correlations are computed from returns, not prices, because two rising
price series correlate whether or not their movements are related.

### The selector learns only from closed trades

Weighting strategies by their measured performance is the most powerful
look-ahead bug available: it would weight by results not yet produced. The
tracker only ingests a trade once it has closed, and refuses to report an
expectancy below a minimum sample size, since two lucky trades are noise rather
than evidence.

### Node failures are contained, not fatal

Every graph node is wrapped so an exception is recorded and the run continues.
The failed component contributes nothing — it is absent, not guessed at. One bad
indicator killing a 3,000-bar run at bar 1,900 loses the run and teaches very
little.

### LIVE mode cannot be selected

`TradingMode.LIVE` exists solely so the validator can reject it with an
explanation. Real-money execution is out of scope, and the kill switch
`trading_enabled` additionally defaults to false so the repo ships disarmed.

---

## Error handling

One failing component must never take down a run. From phase 3, a strategy that
raises is marked `UNAVAILABLE`, the reason is logged, and the graph continues
with the remaining strategies.

Errors that *should* stop a run: corrupt data reaching a strategy (fail closed —
`DataQualityError`), and invalid configuration (fail at startup).

---

## Testing approach

- Default suite is offline and deterministic. Network tests are marked
  `network` and excluded.
- Every fixture writes into a per-test temp directory via `GTP_DATA_ROOT`, so a
  test run can never touch the real dataset.
- Vendor responses are tested against recorded payload shapes, so parsing logic
  is covered without a network dependency.
- Edge cases are first-class: empty datasets, single rows, all-NaN volume,
  duplicate timestamps, inverted OHLC, unparseable dates, expired caches,
  tampered payloads.
