# The Decision Graph

How the LangGraph workflow is assembled, how state flows through it, and how a
failure in one node is contained.

---

## Structure

```
START
  ↓
features
  ↓ (stops here if features could not be computed)
regime
  ↓
┌──────────────────── parallel fan-out ────────────────────┐
│ strategy_trend_following    strategy_breakout            │
│ strategy_support_resistance strategy_mean_reversion      │
│ strategy_momentum                                        │
└──────────────────────────────────────────────────────────┘
  ↓ (fan-in)
selection          weight strategies by regime + measured performance
  ↓
aggregation        combine into one decision
  ↓ (stops here if the ensemble says WAIT)
risk               size the position, or refuse it
  ↓ (stops here if risk refuses)
order              build the order
  ↓
END
```

Render the live structure with `TradingGraph.to_mermaid()`.

## Why LangGraph

It was chosen because it earns its place on three things, not because it is
fashionable: a typed shared state with reducers, genuine parallel fan-out for
the five strategies, and conditional edges that let the run stop early.

The graph is thin. If LangGraph were removed, the nodes, routing functions and
state would survive unchanged — which is the property that made it safe to
adopt.

## Where the graph stops

**The graph never fills an order.** It ends at order creation.

A fill requires the *next* bar's prices, and the graph only ever sees the bar it
is analysing. Letting it fill would bake look-ahead bias into the architecture
itself, where no later test could remove it. The backtester and the paper-trading
loop own the fill, because only they can advance time.

---

## State

One `TradingState` (a `TypedDict`) flows through every node. Nodes return only
the keys they changed.

| Key | Written by |
|---|---|
| `symbol`, `timeframe`, `timestamp`, `market_data`, `equity` | caller |
| `features` | features node, or supplied by the caller |
| `regime` | regime node |
| `strategy_signals` | every strategy node (merged) |
| `strategy_weights` | selection node |
| `aggregated`, `final_signal` | aggregation node |
| `risk_decision` | risk node |
| `order` | order node |
| `errors`, `trace`, `decision` | any node |

### Reducers

`strategy_signals` carries `Annotated[dict, merge_dicts]`. The five strategy
nodes run concurrently and each writes one entry; without a reducer LangGraph
rejects the concurrent update. `errors` and `trace` use an appending reducer for
the same reason.

### The state is the decision record

At the end of a run, everything needed to explain the outcome is present:
regime, every strategy's signal, the weights, the aggregate, the risk verdict,
and any node errors. `explain(state)` renders it:

```
XAUUSD 1D at 2026-08-14 00:00:00+00:00
Regime: TRENDING_UP (confidence 0.81)
Strategies:
  breakout: WAIT
  mean_reversion: WAIT
  momentum: LONG at 0.83, weight 0.34
  support_resistance: WAIT
  trend_following: WAIT
Aggregate: WAIT
  - Market regime TRENDING_UP at 81% confidence
  - momentum: LONG at 0.83 confidence, weight 0.34
  - Winning margin 12% is below the 15% minimum
Decision: WAIT
```

Every line comes from recorded state. Section 66 of the brief requires the
explanation to come from actual state rather than generated prose, and there is
a test asserting the rendered text matches what the state holds.

---

## Error isolation

Every node is wrapped in `@isolated(name)`. A raised exception is caught,
recorded in `state["errors"]`, and the graph continues.

```
Trend Strategy → SUCCESS
Breakout       → SUCCESS
S/R            → ERROR      ← recorded, absent from strategy_signals
Momentum       → SUCCESS
```

The failed strategy contributes no signal — it is **absent**, not guessed at.
The aggregator sees four opinions instead of five and proceeds. A test injects a
deliberately exploding strategy and asserts the run completes, the error is
recorded, and the surviving strategies still produce a decision.

The alternative — one bad indicator killing a 3,000-bar run at bar 1,900 — loses
the run and tells you very little.

---

## Routing

Conditional edges let the run stop early:

| Router | Stops when |
|---|---|
| `route_after_features` | features missing or empty |
| `route_after_aggregation` | the ensemble says WAIT |
| `route_after_risk` | risk refused the trade |

Once the aggregator has said WAIT there is nothing for the risk engine to size,
and running it anyway would produce a meaningless decision record.

---

## Selection and aggregation

### The selector

Weights each strategy for the current bar from two inputs:

1. **Regime fit** — the strategy's own declared preferred and avoided regimes. A
   strategy that declares a regime hostile is **suppressed** there, not merely
   down-weighted. Mean reversion in a strong trend is not a weak opinion, it is
   a wrong one.
2. **Measured performance** — realised expectancy in that regime, from
   `RegimePerformanceTracker`.

Regime confidence scales the whole judgement: an uncertain regime reading should
not drive a confident weighting decision.

**On performance-based weighting and look-ahead:** the tracker learns *only from
closed trades*, so at any bar it reflects strictly past information. It also
refuses to report an expectancy below `min_samples` (default 10) — a strategy
that won its first two trades in a regime is not evidence, and treating it as
such is how a selector overfits to noise inside a single backtest.

### The aggregator

Three methods, all of which can return WAIT:

- `majority` — unweighted vote; a baseline to beat
- `weighted` — votes scaled by weight × confidence (the default)
- `unanimous` — every strategy with an opinion must agree

Plus two gates: a minimum winning margin (default 15%) and a minimum combined
confidence (default 50%).

**Conflicting signals abstain by default.** When strategies genuinely disagree,
that disagreement is information — averaging two opposing views into a third is
not a synthesis, it is noise. `ConflictPolicy.NET_SCORE` will let a lopsided
conflict resolve, and the losing side is still recorded in `opposing`.

**The combined stop is the most conservative of the proposals**, never the
widest. Adopting the widest would silently increase the risk each strategy
believed it was taking.

The result converts to an ordinary `Signal` via `to_signal()`, so the risk
engine consumes the ensemble in exactly the same shape as a single strategy and
stays ignorant of how the decision was reached.

---

## Running the graph directly

```python
from app.graph.workflow import build_graph
from app.portfolio.portfolio import Portfolio

graph = build_graph(config, Portfolio(10_000.0), asset=config.assets.get("XAUUSD"))

state = graph.run(
    symbol="XAUUSD", timeframe="1D", timestamp=ts,
    market_data=data, equity=10_000.0,
    features=features.upto(ts),      # optional; recomputed if omitted
)
print(explain(state))
```

Passing precomputed features matters for the backtester: recomputing them per
bar turns an O(n) run into O(n²) for an identical result. The feature node
detects a supplied set and passes it through untouched.
