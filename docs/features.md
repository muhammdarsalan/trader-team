# Features and Market Structure

What the feature engine computes, and the guarantees it makes.

---

## The two guarantees

**1. Causality.** A feature value at bar `t` uses only bars `<= t`. No rolling
window is centred, no series is shifted backwards, and anything requiring
future confirmation is published only when confirmation arrives.

**2. Honest absence.** A feature that cannot be computed is `NaN`, never a
filled-in guess. Warm-up rows stay `NaN`. Volume features are suppressed
entirely on instruments where volume is meaningless.

Causality is enforced by test, not by inspection. `tests/helpers.py` provides
`assert_causal`, which computes a feature on the first *n* bars, computes it
again on the full series, and compares the overlap. Any difference means the
full computation used future information. Every indicator, every structure
feature and the whole engine end-to-end are covered by it.

```python
def test_indicator_is_causal(name, fn):
    df = make_ohlcv(periods=400)
    assert_causal(fn, df, label=name)   # fails loudly on look-ahead
```

---

## Swing points: the confirmation delay

**This is the part of the platform most likely to be broken by a well-meaning
change.**

A swing high at bar `i` is defined by the bars on *both* sides of it: bar `i` is
a swing high only if the `right` bars after it are lower. That fact is not
knowable at bar `i`. It becomes knowable at bar `i + right`.

The naive implementation marks the pivot at bar `i`. Every strategy reading it
then trades on structure the market had not yet produced, and the backtest
reports superb results that vanish in live trading.

`swing_points()` publishes a pivot at bar `i + right` — the bar where
confirmation actually arrives — while preserving both the pivot's price and its
true age:

| Column | Meaning |
|---|---|
| `swing_high` / `swing_low` | Pivot price, on the confirmation bar only |
| `last_swing_high` / `last_swing_low` | Most recent confirmed pivot, forward-filled. **Strategies use this.** |
| `last_swing_high_age` | Bars since the pivot *occurred* (not since it was published) |
| `prev_swing_high` / `prev_swing_low` | The pivot before the last, for HH/LL comparisons |

With `SwingConfig(left=2, right=2)` and a peak at bar 5:

```
bar 5: swing_high = NaN   ← pivot occurred, not yet knowable
bar 6: swing_high = NaN   ← still unconfirmed
bar 7: swing_high = 20.0  ← published, age = 2
```

---

## Donchian channels exclude the current bar

`donchian_channel(..., exclude_current=True)` shifts the window back one bar.

Without it, "close breaks above the 20-bar high" can never be true at the moment
of the break, because the current bar's own high is inside the maximum it must
exceed. Every breakout would be delayed a bar or lost entirely.

## Breakout freshness

`breakout_up` is true whenever the close is beyond the channel. In a sustained
trend **every bar sets a new N-bar high**, so that flag stays true for hundreds
of consecutive bars.

That is a trend, not a breakout. A breakout is a *transition* out of a range.

`breakout_up_fresh` / `breakout_down_fresh` are true only when the channel was
intact over the preceding `freshness_window` bars. Anything asking "is this a
breakout right now" — the regime detector, the breakout strategy — must use the
fresh variant. This was caught by a test asserting that a clean linear uptrend
is classified `TRENDING_UP`, not `BREAKOUT` forever.

---

## The feature panel

Computed by `FeatureEngine.compute(market_data, asset)`.

### Volatility (computed first — others normalise against ATR)

| Feature | Notes |
|---|---|
| `atr` | Wilder ATR, price units |
| `atr_pct` | ATR relative to price |
| `atr_percentile` | ATR's rank within its own history, `[0, 1]` |
| `realised_vol` | Rolling std of returns |
| `bb_middle` / `bb_upper` / `bb_lower` | Bollinger Bands |
| `bb_width` | Bandwidth relative to the middle band (scale-free) |
| `bb_pct_b` | Position within the bands: 0 at lower, 1 at upper |

### Trend

| Feature | Notes |
|---|---|
| `sma_20` / `sma_50` / `sma_200` | Configurable periods |
| `close_vs_sma_N_atr` | Distance from each SMA **in ATR units** |
| `ema_9` / `ema_21` / `ema_50` | Configurable periods |
| `ma_slope` | Slope of the primary SMA, **normalised by ATR** |
| `adx` / `di_plus` / `di_minus` | Wilder ADX |

### Momentum

`rsi`, `roc`, `momentum`, `macd`, `macd_signal`, `macd_hist`.

### Structure

Swing columns (above), plus `structure` (`UPTREND` / `DOWNTREND` / `UNCLEAR`),
`higher_high` / `lower_high` / `higher_low` / `lower_low`,
`dist_to_resistance_atr`, `dist_to_support_atr`, the Donchian channel,
breakout flags, `channel_width` and `consolidation_ratio`.

### Volume — *conditional*

`volume_sma`, `relative_volume`, `volume_spike`, `volume_percentile`,
`volume_trend`. **Suppressed entirely when `has_reliable_volume: false`.**

---

## Why so much ATR normalisation

Gold trades near 2,000 and EURUSD near 1.08. A raw moving-average slope on the
two differs by three orders of magnitude, so any fixed threshold is meaningless
across instruments — it would fire constantly on one and never on the other.

Dividing by ATR turns "price is far above its average" into "price is 2.3 ATR
above its average", which means the same thing on every instrument and in every
volatility regime. This is why `ma_slope`, `close_vs_sma_N_atr` and the
structure distances are all in ATR units.

The same logic drives `atr_percentile`: "ATR is 12" means nothing on its own;
"ATR is at its 95th percentile" means something everywhere.

---

## Warm-up

`FeatureConfig.warmup_bars()` returns the number of bars needed before every
feature is defined — driven by the longest indicator (typically SMA-200 or the
volatility lookback).

**Strategies must not fire inside the warm-up.** `Strategy.generate_signal()`
enforces this and returns `WAIT` with the reason recorded. An indicator computed
from three observations is not a shorter version of the real one; it is a
different number that will never occur in live trading.

```python
features.warm()                  # rows where everything is defined
features.is_warm_at(timestamp)   # per-bar check
```

---

## Smoothing conventions

RSI, ATR and ADX use **Wilder smoothing** (`alpha = 1/n`), which is what the
original definitions specify and what charting platforms display. Substituting a
standard EMA (`2/(n+1)`) produces visibly different values and silently breaks
comparison against any external reference. There is a test asserting the two
differ, so a future "simplification" to `ewm(span=...)` fails loudly.

---

## Configuration

All periods live in `configs/features.yaml` and are validated into
`FeatureConfig`. They are conventional defaults, **not truths** — every one is a
parameter to be tested for robustness in phase 4. A strategy that works only at
exactly RSI(14) is overfit to a number Wilder chose in 1978.
