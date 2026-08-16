# Backtesting and Execution

How the backtester works, what it assumes, and why every assumption is
deliberately pessimistic.

> **No result produced by this platform is a profitability claim.** A backtest
> describes one historical sample. It is the weakest form of evidence that a
> strategy works, and the machinery that produces stronger evidence —
> out-of-sample splits, walk-forward analysis, Monte Carlo — is phase 4.

---

## Bar ordering: the correctness argument

Everything hinges on what happens in what order within a single bar.

```
For bar i:
  1. Fill orders created at bar i-1, at bar i's OPEN
  2. Check exits for open positions against bar i's HIGH and LOW
  3. Mark to market at bar i's CLOSE, record the equity snapshot
  4. Run the decision graph on data up to and including bar i
     → any order it creates is queued for bar i+1
```

Step 4 running *after* step 3, and its order filling at step 1 of the **next**
bar, is what makes the backtest honest. A signal derived from bar `i`'s close
cannot be filled at that close: by the time the decision exists, that price has
already traded.

A position entered at a bar's open can also exit on that same bar. That is not
an edge case to be smoothed away — a gap straight through the stop is exactly
how real trades die.

### Causality is verified, not asserted

```python
def test_backtest_is_causal(config):
    full      = Backtester(...).run(market(df))
    truncated = Backtester(...).run(market(df.iloc[:500]))
    # Trades before the cutoff must be identical in both runs.
```

If running on the first 500 bars produces different trades from running on all
700 and looking at the first 500, the full run consumed information that had
not happened yet. The same test exists for the equity curve.

This sits on top of the phase-2 `assert_causal` suite, which proves every
feature is causal independently. That proof is what makes it safe to compute
features **once** over the whole series rather than recomputing them per bar —
an O(n²) cost for an identical answer. Each bar receives `features.upto(ts)`, a
truncated view, so the guarantee is structural as well as tested: a strategy
cannot read a row that is not in the frame it was handed.

---

## Execution assumptions

Every default is pessimistic, because optimistic execution manufactures profit
that does not exist. Configured in `configs/execution.yaml`.

| Assumption | Default | Why |
|---|---|---|
| Fill delay | next bar's open | The signal bar's close has already traded |
| Spread | buy the ask, sell the bid | Paid on entry **and** exit |
| Slippage | 0.05 × ATR | Fills are worst when markets move fastest |
| Commission | configurable | Per-unit, percentage, and a minimum |
| Gaps | honoured | A gap through the stop fills at the open |
| Same-bar ambiguity | `stop_first` | See below |

### The same-bar problem

When one bar's range contains both the stop and the target, **OHLC data cannot
say which was touched first.** Resolving it truthfully needs intrabar data the
platform does not have.

The three options:

- `stop_first` — assume the loss. **The default.**
- `target_first` — assume the win. Believing the flattering possibility every
  single time; never use it for research.
- `skip` — discard such trades and report the count.

The metrics report surfaces the count of ambiguous exits so the reader knows how
much of the result rests on this choice.

### Gaps

If a bar opens beyond the stop, the fill is at the **open**, not at the stop
price. Pretending otherwise deletes exactly the losses that hurt most — the ones
that arrive overnight with no chance to react.

The same applies at entry: if the market gapped past the stop before the queued
order filled, the trade is abandoned rather than opened already beyond its own
stop.

---

## Position sizing

The arithmetic is deliberately simple enough to check by hand:

```
risk_amount   = equity × risk_per_trade      # 10,000 × 1%  = 100
risk_per_unit = |entry − stop|               # |3000 − 2990| = 10
quantity      = risk_amount / risk_per_unit  # 100 / 10      = 10 units
```

Every other rule is a **cap** applied on top, and caps only ever shrink a
position.

### Risk sizing can imply leverage nobody chose

Those 10 units of $3,000 gold are $30,000 of notional against $10,000 of
equity — 3× leverage, arrived at without anyone deciding to use leverage. The
per-position notional cap (default 50% of equity) is what prevents risk-based
sizing from quietly implying it, and on high-priced instruments it binds before
the risk calculation does. There is a test pinning exactly this interaction.

### Limits

| Limit | Default | Blocks or reduces |
|---|---|---|
| `risk_per_trade` | 1% | sizing input |
| `max_position_notional_pct` | 50% | reduces |
| `max_concurrent_positions` | 3 | blocks |
| `max_positions_per_symbol` | 1 | blocks |
| `max_portfolio_risk` | 6% | reduces, then blocks |
| `max_risk_per_strategy` | 3% | reduces, then blocks |
| `max_correlated_risk` | 3% | reduces, then blocks |
| `drawdown_reduce_threshold` | 10% | halves size |
| `max_drawdown_limit` | 20% | blocks |
| `daily_loss_limit` | 3% | blocks for the day |

Order matters: the daily-loss limit is checked before drawdown scaling, so a
loss taken today blocks trading outright rather than merely shrinking the next
position.

### Correlation is exposure

Three "independent" longs in EURUSD, GBPUSD and AUDUSD are one short-dollar bet
wearing three hats. Sizing them as independent triples the real risk.

Positions in symbols correlated above `correlation_threshold` share a single
risk budget. Negative correlation counts too: short EURUSD and long USDCHF is
one trade. Correlations are computed from **returns**, not prices — two rising
price series correlate whether or not their movements have anything to do with
each other.

### The kill switch

`trading_enabled` defaults to **false** and ships that way. While false the
graph still runs and still analyses; every blocked order is recorded with reason
`KILL_SWITCH`. The backtester passes `trading_enabled=True` explicitly, because
a backtest that opens nothing is not a backtest.

### Data quality is a risk limit, not a report header

Every input to position sizing — equity, stop distance, ATR — descends from the
price series. So the grade the quality gate gave that series reaches the sizing
decision:

| Grade | Default behaviour | Reason |
|---|---|---|
| `FAIL` | Rejected, reason `DATA_QUALITY` | No size computed from corrupt inputs is meaningful |
| unstated | Rejected, reason `DATA_QUALITY_UNKNOWN` | "Nobody checked" must not size the same position as "checked and passed" |
| `WARNING` | Allowed, recorded in the decision metrics | Nearly every long window has a stale final bar or a holiday gap |
| `PASS` | Allowed | — |

The literal string `"UNKNOWN"` is *not* a grade: it normalises to the same
unverified state as no answer at all, so a placeholder cannot be mistaken for a
pass.

`MarketDataService` already refuses to hand over FAIL-grade data, but that is
not the only route in — a caller can disable validation or build a `MarketData`
straight from a frame. So `Backtester.run` **grades the series itself** when no
grade is supplied, and provenance records whether the grade came from the
`caller`, was `self-graded`, or where grading failed. Repeating the check where
money is committed means a degraded series cannot reach a position size by
taking a side entrance.

For paper trading, where a stale or gappy feed is a live hazard rather than a
footnote, set `block_on_data_quality_warning: true` in `configs/risk.yaml`.

---

## Metrics

Computed on every run. The full list is in `app/backtest/metrics.py`.

**Win rate is the most misleading number available.** A 90% win rate with a 20:1
loss ratio loses money; a 30% win rate with 5:1 winners makes it. There is a
test asserting exactly this. The numbers that decide whether a strategy is worth
anything are **expectancy** (per trade, and in R) and **risk-adjusted return**
(Sharpe, Sortino, Calmar).

### Warnings travel with the numbers

The metrics object carries a `warnings` list, rendered alongside the report
rather than buried:

- Fewer than 30 trades — nothing here is stable
- Infinite profit factor — no losing trades means a tiny sample, not perfection
- Win rate > 75% on < 100 trades — check for look-ahead bias
- Large return with almost no drawdown — the signature of a bug
- Sharpe > 3 — far above what systematic strategies sustain out of sample
- Ambiguous same-bar exits — how many trades rest on that assumption
- Trades refused because of the window's data quality — few trades then read as
  a refusal rather than as a quiet strategy

These sit next to the figures because the figures are what get copied into a
decision.

---

## Reproducibility

Every run records what produced it:

```json
{
  "experiment_id": "EXP-2026-001",
  "symbol": "XAUUSD", "timeframe": "1D",
  "start": "2015-01-02", "end": "2026-08-14", "bars": 2920,
  "data_provider": "yahoo",
  "data_checksum": "8efec84040b805bf3e45e41ec2b5c848",
  "data_quality": "WARNING",
  "data_quality_source": "caller",
  "git_revision": "1207966-dirty",
  "random_seed": 42,
  "config_snapshot": { "risk": {...}, "execution": {...}, ... }
}
```

`git_revision` records `-dirty` when the working tree has uncommitted changes,
because a result produced from modified code is not reproducible from the commit
alone and saying so beats a hash that lies.

`data_quality` travels with the result, so a figure computed on WARNING-grade
data says so on its face. `data_quality_source` says where that grade came from
— `caller`, `self-graded` or `grading-failed` — because a result is only as
trustworthy as the provenance of the check behind it.

For studies rather than single runs, phase 4 adds a SQLite experiment store with
ids derived from the configuration, the data and the code rather than from a
counter. See [research.md](research.md).

```bash
python scripts/run_backtest.py --symbol XAUUSD --timeframe 1D --start 2012-01-01 --save
```

Writes `experiments/EXP-YYYY-NNN/` containing `provenance.json`, `metrics.json`,
`report.txt`, `trades.csv`, `equity_curve.csv`, `decisions.csv` and
`regime_performance.csv`.

---

## The trade log

Every completed trade records entry and exit time and price, direction, size,
stop, target, strategy, entry and exit regime, gross P&L, costs, net P&L, R
multiple, bars held, MAE and MFE, and the exit reason.

MAE and MFE (maximum adverse and favourable excursion) are tracked from the
bar's high and low, not its close, so "how far underwater did this trade go"
has a truthful answer.

## The no-trade log

`decisions.csv` records **every analysed bar**, including the ones that produced
nothing — which is the majority. Section 67 of the brief requires the system to
record why it did not trade, and the aggregate appears in the report:

```
NO-TRADE DECISIONS
  WAIT                                 1,450  (53.3%)
  BLOCKED_MAX_POSITIONS_PER_SYMBOL     1,075  (39.5%)
  ORDER_LONG                             136  (5.0%)
  ORDER_SHORT                             59  (2.2%)
```

Knowing *why* nothing happened is what distinguishes "the strategy saw nothing"
from "the strategy was broken and produced nothing".
