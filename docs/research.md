# Research and validation

Phase 3 could answer *"what would this configuration have done on this
history?"*. That is the weakest question in the project, and a good answer to it
is close to worthless: the configuration exists **because** of that history.

This phase asks the harder questions.

- Does it still work on data it was never developed on?
- Does it keep working as the window rolls forward through time?
- Does the result survive the trades arriving in a different order?
- Does it survive small changes to parameters nobody has any reason to believe
  are exactly right?
- How much of what we are looking at is explained by how many things we tried?

**Nothing in this phase optimises toward a return, and nothing in it concludes
that a configuration is profitable.** The output is evidence and its limits,
including the frequent and entirely legitimate conclusion that a configuration
has not been shown to work.

---

## Running a study

```bash
python scripts/run_research.py --symbol XAUUSD --timeframe 1D --start 2012-01-01
```

Faster, for iterating:

```bash
python scripts/run_research.py --symbol XAUUSD --no-robustness --simulations 500
```

What has been tried already:

```bash
python scripts/run_research.py --list-experiments
```

In code:

```python
from app.config.loader import get_config
from app.data.service import MarketDataService
from app.research.experiments import ExperimentStore
from app.research.study import ValidationStudy

config = get_config()
requested = MarketDataService().get_historical_data("XAUUSD", "1D", start="2012-01-01")

study = ValidationStudy(
    config=config,
    data=requested.data,
    asset=config.assets.get("XAUUSD"),
    quality_status=str(requested.quality.status),
    store=ExperimentStore(),
)
result = study.run()
print(result.report.render())
```

Reports land in `reports/validation/<experiment-id>/`; metadata goes to
`experiments/experiments.db`.

---

## The order of operations

The sequence is the design, not a convenience:

1. Split the history into in-sample, validation and out-of-sample windows.
2. Run the baseline on **in-sample** and **validation**.
3. Sweep parameter neighbourhoods — on the in-sample window, because a sweep is
   a search and a search consumes whatever it touches.
4. Measure strategy correlation and turn any redundancy into candidate
   configurations. Nothing is removed.
5. Walk forward, choosing between candidates on each fold's training half only.
6. Run the **out-of-sample** window. Once. Last.
7. Resample the out-of-sample trade sequence.
8. Count how much searching produced all of this, and assess it.
9. Write the report and record everything.

Step 6 happens after every decision has already been made. A decision informed
by the final window would make that window in-sample without anybody noticing,
and the output would look identical either way.

---

## Temporal separation

`app/research/splits.py`. Three details make a split honest.

**Chronological, never random.** Randomly shuffling rows into train and test
sets destroys the property entirely on a time series: a randomly chosen "test"
bar sits between two "training" bars whose prices all but determine it. The
out-of-sample window is the *last* window, because research proceeds forward
through time.

**Warm-up prefixes are not evaluation.** A feature needs history before it is
defined, so each segment carries a read-only prefix of preceding bars. Reading
price history that precedes a decision is not look-ahead — that is simply what a
trader has. Reading anything after it is, and no prefix ever extends forward.
`Segment` refuses to construct one that does.

**Segments are separated by an embargo.** A trade opened near the end of one
window can still be open at the start of the next. Without a gap, the same
market episode is scored twice and the second scoring is contaminated by a
position the first one opened. `embargo_bars` in `configs/research.yaml`
controls it; the default is 5.

`assert_temporal_separation` checks the windows are ordered and disjoint, and
the split constructors call it themselves rather than leaving it to a test to
remember.

---

## Walk-forward

`app/research/walk_forward.py`. Decide using only what had happened by date *t*,
trade the window that followed, roll forward, repeat. A single out-of-sample
split answers "did this survive one unseen period?" once, and one good answer is
obtainable by luck; a run of good answers is much harder to get by luck.

The selection step is where walk-forward earns its name and where it is most
often quietly broken. When candidates are supplied, **the choice for each fold
is made from that fold's training window alone**. Choosing once, globally, by
looking at the whole series and then "walking forward" with the winner turns the
analysis into an elaborate in-sample result. The test window is not run until
after the choice is fixed, and `test_selection_uses_only_the_training_window`
pins it.

Two summary numbers:

| Number | Meaning |
|---|---|
| **Walk-forward efficiency** | Mean test performance over mean training performance. Always below 1.0; the question is how far. |
| **Fold consistency** | How many folds were positive. A total carried by one window is a result about that window. |

Rolling windows stay a fixed length; anchored windows grow from a fixed start.
Neither is correct in general, which is why the choice is recorded with the
result.

---

## Monte Carlo

`app/research/monte_carlo.py`. A backtest reports one path: the trades in the
order history happened to deliver them. That order is not a property of the
strategy. Reshuffle the same trades and the return is identical while the
drawdown — the number that decides whether an account survives — can double.

| Method | Question it answers | What it cannot say |
|---|---|---|
| **Permutation** | Given this set of outcomes, how much of the drawdown was luck of sequencing? | Anything about outcomes the sample did not contain. |
| **Bootstrap** | If these trades were draws from a stable process, what else could it produce? | Whether that premise holds. It is a premise, not a fact. |

`block_size` above 1 resamples in contiguous blocks, preserving whatever serial
dependence exists between consecutive trades. Losing streaks cluster in trending
systems, and a resampling that breaks the clusters understates the drawdown it
is supposed to be measuring.

P&L is applied additively in the currency it was earned in. Real sizing is a
fraction of equity, so an early losing streak would shrink later trades; the
fixed-amount assumption slightly overstates the damage of a deep drawdown. It is
the more pessimistic of the two available assumptions, which is why it is the
one used.

**What this cannot do:** no resampling of past trades says whether the strategy
will keep producing trades like these. It bounds sequencing risk, not model
risk, and the two are routinely confused.

---

## Parameter robustness

`app/research/robustness.py`. Every threshold in `configs/` is a guess. RSI(14)
is a number Wilder picked in 1978; a 2.0 ATR stop is round because humans like
round numbers.

The test is **not** "which value is best". That question is the overfitting
machine itself. The test is **is the neighbourhood flat?** A parameter whose
neighbouring values all behave similarly is describing something real, however
mediocre. A parameter with a sharp peak at the shipped value is describing the
sample.

Two consequences, both implemented:

- The report never recommends the best value found. It reports the shape.
- A peak **at the current setting** is a red flag, not confirmation — that is
  exactly the shape produced by having tuned it.

A parameter is called *fragile* when the objective varies more than its own mean
across the neighbourhood, when one step moves it far more than the others do, or
when the shipped value is the local maximum.

Parameters are addressed by dotted path, and a sweep that would produce an
invalid configuration raises instead of quietly producing a data point:

```yaml
robustness:
  parameters:
    - features.rsi_period
    - risk.risk_per_trade
    - strategies.strategies.trend_following.params.adx_minimum
```

---

## Overfitting and data snooping

`app/research/overfitting.py`. The uncomfortable arithmetic: test twenty random
configurations on the same history and the best of them has an expected Sharpe
well above zero **even when none has any edge**. That number is not a discovery;
it is the maximum of twenty draws from a distribution centred on nothing.

So the question is never "is this Sharpe good?" but "is it better than the best I
would expect by chance, given how many things I tried?".

- **Expected maximum Sharpe under the null** — what the best of *N* worthless
  strategies scores, from the Gumbel approximation to the maximum of independent
  draws.
- **Deflated Sharpe ratio** — the probability the observed Sharpe exceeds that
  benchmark after adjusting for the trial count, the sample length, and the skew
  and fat tails of the returns. Trading returns are negatively skewed and
  heavy-tailed; both flatter a naive Sharpe.

Neither is a verdict. A high deflated Sharpe means one specific way of being
fooled has been checked for.

Findings are graded `INFO` / `CAUTION` / `SEVERE`. A severe finding blocks a
positive verdict.

### Counting the trials honestly

The trial count comes from the SQLite store, not from the current session, and
this is the point: **last week's twelve variants still happened.** The expected
best result of a search does not reset when the process restarts. Trials are
counted per distinct configuration against a given data checksum — the same
configuration evaluated twice is one trial, because overstating the search
distorts the arithmetic as badly as understating it.

Running with `--no-store` makes the count cover the current session only, which
understates it. The report says so.

---

## Strategy correlation

`app/research/correlation.py`. Five strategies are not five independent opinions
if three of them read the same moving average. When they agree, the agreement
carries no extra information. When they lose, they lose together, and the
portfolio risk limits — which assume the positions are separate bets — are sized
against a diversification that does not exist.

Two kinds of correlation, answering different questions:

- **Return correlation** — do they make and lose money at the same times? This
  is what matters for portfolio risk. Because the graph aggregates every
  strategy into one signal before a position opens, the trade log records the
  ensemble rather than which strategy earned what, so this is measured by
  running each strategy alone over the same window.
- **Signal agreement** — do they say the same thing at the same bars, including
  the bars where neither trades? Agreeing to wait is agreement.

### Redundancy is never acted on automatically

**A high correlation is not evidence that removing a strategy improves
anything.** The two may be redundant on this sample and diverge on the next; the
"redundant" one may be the one that survives; and dropping a strategy because of
a correlation measured on the same data used to justify keeping the others is a
decision made entirely in-sample. Deleting components on the strength of an
in-sample correlation matrix is a well-travelled route to a system beautifully
diversified against the past.

So each finding becomes a **hypothesis** plus a candidate configuration:

| Variant | What it tests |
|---|---|
| `halve_<strategy>` | Standing weight reduced to 0.5 (`StrategyConfig.weight`) |
| `drop_<strategy>` | Strategy disabled entirely |

These are run on the **validation** window — not the window that suggested them,
and not the out-of-sample window, where comparing several configurations and
reporting the best would consume the one result that is supposed to mean
something. The comparison reports the change in expectancy **and** the change in
drawdown, and calls a variant that improves one while worsening the other a
trade-off rather than an improvement.

The configuration is left unchanged unless a walk-forward run agrees.

---

## Experiment tracking

`app/research/experiments.py`, backed by SQLite at
`experiments/experiments.db` (gitignored — it is derived data, and committing it
would put one machine's search history into everyone else's overfitting
arithmetic).

### Reproducible identity

An experiment id is derived from what actually determines the result:

```
RES-<sha256(config, data checksum, code revision, seed, study design)[:12]>
```

Run the same study twice and you get the same id and one row, not two rows that
look like independent confirmations. Change one parameter and you get a
different id automatically. Wall-clock time, machine and user stay out of the
hash; Python's own `hash()` is salted per process and would defeat the whole
point, so `stable_hash` uses SHA-256 over canonical JSON.

### Tables

| Table | Holds |
|---|---|
| `experiments` | One row per study: provenance, config snapshot, study design, verdict |
| `segments` | Each window, its dates, trade count and data-quality grade |
| `metrics` | Every numeric metric per window (non-numeric and infinite values are skipped) |
| `findings` | Overfitting diagnostics with severity and evidence |
| `trials` | Every configuration evaluated against a data checksum — the data-snooping counter |
| `artifacts` | Where the report was written |

---

## The report

`app/research/report.py`. The point is that someone can read it and **disagree**
with it. A single headline number cannot be argued with, only believed or not.

Sections: how to read it · in-sample / validation / out-of-sample · walk-forward
· drawdown behaviour · Monte Carlo · parameter sensitivity · strategy
correlation and weight variants · performance by market regime · overfitting and
data snooping · verdict.

Drawdown gets its own section because a return is an outcome while a drawdown is
an experience, and the experience decides whether a system is still being run
when the outcome arrives.

### Verdicts

| Verdict | Meaning |
|---|---|
| `INSUFFICIENT EVIDENCE` | Too few out-of-sample trades to conclude anything |
| `EVIDENCE AGAINST` | Negative out-of-sample expectancy. **A successful research outcome** |
| `INCONCLUSIVE` | A severe diagnostic, inconsistent folds or a fragile parameter |
| `SURVIVED THIS ROUND OF TESTING` | The tests run did not break it |

There is no verdict meaning "profitable", and there never will be.
`SURVIVED THIS ROUND OF TESTING` is the weakest positive result available: it
means the tests that were run did not break this configuration. Other tests
exist, this is one instrument, and the future is not a sample from the past.

---

## Data quality reaches the risk engine

Position sizing is arithmetic on the price series — equity, stop distance and
ATR all descend from it. So the grade the quality gate gave that series has to
reach the sizing decision rather than stopping at the report header.

The gate is enforced in three places:

1. **`MarketDataService`** refuses to hand over FAIL-grade data (phase 1).
2. **`Backtester.run`** grades the series itself when no grade is supplied, so
   reaching the backtester directly does not skip the check. Provenance records
   whether the grade was `caller`-supplied or `self-graded`.
3. **`RiskEngine.evaluate`** refuses to size a position on FAIL-grade data, and
   refuses on an *unstated* grade — "nobody checked" must not produce the same
   position size as "checked and passed".

Each research segment is graded on **its own** data, because a grade for a decade
says nothing about which year inside it was gappy. A window whose grade blocks
trading reports that explicitly, so few trades read as a refusal rather than as a
quiet strategy.

```yaml
# configs/risk.yaml
block_on_data_quality_fail: true       # FAIL never sizes a position
require_known_data_quality: true       # an unstated grade is a refusal
block_on_data_quality_warning: false   # turn on for paper trading
```

WARNING-grade data still trades by default: nearly every long historical window
carries at least one warning (a stale final bar, a holiday gap), and blocking all
of them would block all research. For paper trading, where a stale feed is a live
hazard rather than a footnote, turn it on.

---

## Configuration

Everything lives in [`configs/research.yaml`](../configs/research.yaml). None of
it is a performance dial — every setting decides how hard a configuration is
tested, and loosening one makes the result look better without making it truer.

| Setting | Purpose |
|---|---|
| `split_fractions` | In-sample / validation / out-of-sample proportions |
| `embargo_bars` | Gap between windows so trades cannot straddle a boundary |
| `objective` | How candidates are ranked. Return-maximising objectives are refused |
| `min_trades_for_conclusions` | Below this, the study declines to conclude |
| `correlation_threshold` | Above this, a pair becomes a redundancy hypothesis |
| `walk_forward.*` | Fold count, window sizes, anchored or rolling |
| `monte_carlo.*` | Simulations, method, block size |
| `robustness.*` | Which parameters to sweep, and how widely |

### Why return is not a selectable objective

`get_objective("return")` raises. Ranking configurations by return rewards
whichever one happened to size up before the biggest move in the sample; it has
no risk in it at all, and picking by it reliably selects the variant that took
the most risk rather than the one that earned the most per unit of it. The
available objectives are `sortino`, `sharpe`, `expectancy_r`, `calmar` and
`risk_adjusted`, and each refuses to rank a sample below 20 trades — a
configuration that took four trades and won three has an excellent everything.

---

## What this phase does not establish

- That any configuration is profitable, now or ever.
- That out-of-sample survival predicts future survival. It is evidence, and weak
  evidence at that.
- That the execution assumptions are right. They are estimates, and optimistic
  execution is the most common way a strategy looks good on paper and is not in
  practice.
- That results on one instrument transfer to another.
- That a flat parameter neighbourhood means the parameter is *right*. It means
  the result is not an artefact of that exact number.

The most likely correct conclusion from any given study is that the
configuration under test has not been shown to work. Reaching that conclusion
cheaply is what this phase is for.
