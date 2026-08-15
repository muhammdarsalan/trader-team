# Strategies, Regimes and Signals

The five strategies, the regime detector they consume, and the contract they
all return.

> **No strategy here is known to be profitable.** Phases 1–2 deliberately make
> no performance claims whatsoever. Every parameter is a hypothesis, and the
> platform exists to test them in phase 4 — including the possibility that all
> five fail, which is a successful research outcome.

---

## The Signal contract

Every strategy returns `app.signals.models.Signal`. No strategy invents its own
output shape: the moment two disagree, the aggregator, risk engine and
backtester all need special cases, and adding a sixth means touching five files.

```python
Signal(
    strategy="trend_following",
    symbol="XAUUSD",
    timeframe="1D",
    direction=SignalDirection.LONG,   # LONG | SHORT | WAIT
    confidence=0.76,                  # [0, 1]
    timestamp=...,
    entry_price=...,                  # the signal bar's CLOSE
    stop_loss=...,                    # required for actionable signals
    take_profit=...,                  # optional
    reasoning=["ADX 31.2 confirms trend strength", ...],
    metadata={"adx": 31.2, ...},
)
```

### Self-validation

The object validates on construction and refuses to exist in an invalid state:

- Long stops must be **below** entry; short stops **above**. A stop on the wrong
  side produces a negative risk distance and would blow up the risk engine three
  layers downstream, where the cause is unrecognisable.
- Targets must be on the profitable side.
- Actionable signals **require a stop**. Without one the risk engine cannot size
  the position, and an unsized position is an uncapped loss.
- Prices must be positive; confidence must be in `[0, 1]`.
- Error messages name the strategy, because five of them are running.

### What confidence is not

Confidence is the strategy's own assessment of **how well its conditions were
met**. It is *not* a probability of profit, is not calibrated against outcomes,
and nothing in the platform treats it as if it were. A signal at 0.9 confidence
is not "90% likely to win" — it means the setup was textbook.

### WAIT is a real answer

`WAIT` is the correct answer most of the time, and every one records *why*:

```python
Signal.wait("mean_reversion", "XAUUSD", "1D",
            reasoning=["ADX 34.1 exceeds the 20 ceiling - a trend is present"])
```

Recording why nothing happened is what distinguishes "the strategy saw nothing"
from "the strategy was broken and produced nothing".

---

## Market regimes

`RegimeDetector` classifies each bar into `TRENDING_UP`, `TRENDING_DOWN`,
`RANGING`, `HIGH_VOLATILITY`, `LOW_VOLATILITY`, `BREAKOUT` or `UNCERTAIN`.

Deliberately **rule-based**. Everything it concludes traces to a number you can
print, so a wrong classification can be debugged rather than shrugged at. An ML
classifier fits behind the same interface later; starting with one would mean
never being able to tell a modelling bug from a market that simply changed.

### How it decides

Three independent directional votes must agree:

1. **DI spread** — `di_plus` vs `di_minus` from ADX.
2. **Normalised MA slope** — in ATR per bar, so one threshold works everywhere.
3. **Confirmed swing structure** — `UPTREND` / `DOWNTREND` / `UNCLEAR`.

Then ADX decides strength: above `adx_trending` (25) is a trend, below
`adx_ranging` (20) is a range. **Between the two is a deliberate no-man's-land**
that yields `UNCERTAIN` rather than forcing a label.

Breakouts take precedence over trends — but only *fresh* ones (see
[features.md](features.md#breakout-freshness)). Volatility extremes are reported
in preference to a bare `RANGING`, because they change how a position should be
sized and stopped.

### UNCERTAIN is a first-class answer

Most of the time the market is not doing anything cleanly classifiable. A
detector that always picks a confident label is lying at least half the time.
Below `min_confidence` (0.45) the reading is withheld and the reason recorded.

### Reasoning comes from state, never from decoration

```
Regime:         TRENDING_UP
Confidence:     0.82
Volatility:     MEDIUM
Trend strength: 0.71
Reasoning:
  - ADX 35.6 is above the trending threshold 25
  - Moving-average slope +0.184 ATR/bar
  - Confirmed swing structure is UPTREND
  - 100% of directional indicators agree
```

A test asserts the ADX quoted in the text matches the metric actually recorded,
so the explanation cannot drift into fiction.

---

## The five strategies

Each declares its own `StrategyMetadata`: supported timeframes, minimum history,
indicators used, preferred and avoided regimes, and **its assumptions**.

### 1. Trend following

*Hypothesis: an established directional move continues more often than it
reverses within the holding period.*

Fast EMA above slow EMA, price on the correct side of the 200-period filter,
ADX above a minimum. Stop and target are ATR multiples.

**Expect a low win rate.** Trend following survives on the size of its winners;
judging it by win rate produces exactly the wrong conclusion. It is whipsawed in
ranging markets, which it declares as an avoided regime.

### 2. Support / resistance

*Hypothesis: price reacts at levels where it previously reversed.*

Uses **only confirmed swing pivots**, so it cannot trade a level the market had
not yet established. Requires price within `proximity_atr` of the level, the
level to be neither too fresh nor stale, and a rejection wick of at least
`rejection_wick_ratio` of the bar's range. The stop goes *beyond* the level — if
the level breaks, the premise of the trade is gone.

**Honest caveat:** support/resistance is the most subjective idea in technical
analysis, and a level drawn after the fact always looks predictive. The
confirmation delay is what keeps this one honest.

### 3. Breakout

*Hypothesis: when price leaves a consolidation, the move continues far enough to
pay for the false breaks.*

Requires a **fresh** break of the Donchian channel, a close clearing the level by
`buffer_atr`, and a prior tightening range. Volume confirmation is optional and
automatically skipped where volume is meaningless.

**Most breakouts fail.** The edge, if any, depends entirely on the size of those
that do not.

### 4. Mean reversion

*Hypothesis: after an extreme move away from a short-term average, price returns
toward it.*

Price outside a Bollinger band, RSI at an extreme, price stretched at least
`min_distance_atr` from the mean. The target **is the mean** — that is the whole
thesis, not an arbitrary ATR multiple.

**The ADX gate is not optional.** Fading a strong trend loses money
indefinitely: "oversold" in a downtrend means the downtrend is working. The
strategy refuses to trade when ADX indicates a trend *and* when the regime is
`TRENDING_*`. There is a test asserting it never fires in a confirmed trend.

Expect a **high win rate with small winners** — the mirror image of trend
following. One trend can erase many wins. Only expectancy settles it.

### 5. Momentum

*Hypothesis: strong recent rate of change persists.*

Rate of change beyond a threshold, RSI on the correct side of the midline,
optional MACD agreement.

**Momentum and trend following correlate heavily.** That correlation is itself a
research question for phase 4: five strategies that all lose in the same week
are not a diversified portfolio, they are one strategy wearing five hats. The
distinction is that trend following waits for *structural* MA alignment while
momentum reacts to the *rate* of change — they disagree most visibly at turning
points.

---

## Look-ahead safety

Three layers, all enforced by test:

1. **Features** are causal (`assert_causal` on every indicator and the engine).
2. **Swing structure** is published only at confirmation.
3. **Signals** are pinned: a test asserts the signal for bar `t` is byte-identical
   whether the series ends at `t` or continues for another hundred bars —
   direction, confidence, entry, stop and target all compared.

Entry price is always the signal bar's **close**, the last price actually
knowable when the decision is made. The backtester (phase 3) fills at the *next*
bar's open plus costs; it never fills at the signal price.

---

## Error isolation

`generate_signal()` returns `WAIT` rather than raising when preconditions fail:
insufficient history, unsupported timeframe, inside warm-up, missing indicators.
One strategy declining to trade — or being misconfigured — cannot interrupt the
other four. A test runs every strategy across hundreds of bars of three
different market shapes asserting nothing raises.

---

## Adding a strategy

1. Subclass `Strategy` in `app/strategies/your_strategy.py`.
2. Declare `StrategyMetadata`, including honest assumptions.
3. Implement `_generate(ctx) -> Signal`.
4. Register it in `app/strategies/__init__.py`.
5. Add a block to `configs/strategies.yaml`.

Nothing else changes — not the graph, the backtester, the risk engine or the
dashboard.

```python
class MyStrategy(Strategy):
    metadata = StrategyMetadata(
        name="my_strategy",
        description="...",
        supported_timeframes=("4H", "1D"),
        min_history_bars=200,
        indicators_used=("rsi", "atr"),
        assumptions=("states what must be true for this to work",),
    )

    def _generate(self, ctx: StrategyContext) -> Signal:
        row = ctx.row
        if some_condition_not_met:
            return self._wait(ctx, "why nothing happened")
        stop, atr = self._atr_stop(ctx, SignalDirection.LONG, 2.0)
        return self._build_signal(
            ctx, SignalDirection.LONG, confidence, stop, target, reasoning
        )
```

Helpers available: `self._wait(...)`, `self._build_signal(...)`,
`self._atr_stop(...)`, `self.param(key, default)`. Confidence floors and
precondition checks are applied by the base class, so a new strategy cannot
forget them.

---

## Configuration

`configs/strategies.yaml` controls which strategies run and with what
parameters. `configs/regimes.yaml` holds the classification thresholds — the
ADX 20/25 split is convention, not law, and phase 4 should test it rather than
inherit it.
