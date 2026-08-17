# Autonomous improvement journal

One entry per change made while the search runs. Every entry was verified with
`python tests/test_core.py` (35/35) before the searcher was restarted onto it.

Newest first.

---

## 2026-08-17 — tick 7 · the HTML now tells you when something is real

Requested: highlight a genuinely good find instead of leaving it as row 1 of a
list that always has a row 1.

**`metrics.assess()` — a seven-point checklist** (`nqq/metrics.py`)
    edge survives costs        expectancy > 0 after the round-trip cost
    beats the search's luck    t_stat >= noise_ceiling(n_tried) + 0.50
    reward/risk >= 2.0         realised payoff, not the tp/sl geometry
    enough trades              >= max(min_trades, 200)
    works in most folds        >= 65% of walk-forward folds profitable
    not one lucky fold         best fold <= 45% of total R
    drawdown is survivable     recovery factor >= 2.0
plus an eighth, the frozen holdout, when it has been run.

`noise_ceiling(n)` is the expected maximum of n standard normals — the best
t-stat a search of this size produces from pure noise. It RISES as the search
runs (2.51 at 100 genomes, 3.04 at 1000, 3.34 at 5000), so the bar tightens
automatically as more lottery tickets are bought. Nothing else in the system
adjusts for having looked at a thousand strategies.

**Tiers:** `candidate` passes all seven; `confirmed` also survived the holdout;
`watch` misses exactly one; anything else shows a bare `4/7`. Failing the
holdout drops the tier to nothing regardless of the other seven — it is the one
test the search cannot game, because it cannot see the data.

**Rendering** (`nqq/report.py`): a green banner above the leaderboard naming the
find, why it qualified, and the exact next command; a `verdict` column; tinted
rows with a coloured left edge; and on the per-strategy page the full checklist
with each item passed or failed and the number behind it.

**`tools/holdout.py` writes its result back to the registry**, so a strategy can
actually be promoted to `confirmed` on the leaderboard rather than the verdict
living only in a terminal that has since scrolled away.

Verified by rendering a synthetic qualifying strategy and reading the page back.
Current real state: **nothing qualifies** — the best live strategy passes 5/7,
failing "beats the search's luck" (t 2.07 vs 3.67 needed at 1,205 tried) and
usually reward/risk. That is the correct answer, and it is the first version of
this leaderboard that says so out loud.

---

## 2026-08-17 — tick 6 · data/ folder

Files dropped into `data/` are picked up automatically: delimiter sniffed,
headerless files handled, overlapping bars de-duplicated, empty `Volume`
replaced by tick counts, and files coarser than `data.timeframe` skipped with a
warning rather than silently blended into a series that looks fine and is
nonsense.

**The bar cache is now keyed on file name, size and mtime** (`nqq/data.py`).
It was keyed on config values alone — and dropping a file into a folder changes
no config value, so you could have added a year of data, seen "bars from cache",
and searched the old bars indefinitely.

Five new tests cover the contract (40 total). Writing them caught a real bug:
`_median_spacing_minutes` assumed nanosecond index resolution, but this pandas
defaults to microseconds, so a daily file measured as 1.44 minutes apart and
sailed straight through the coarse-file guard.

---

## 2026-08-17 — tick 5 · the old leaderboard did not survive the wider window

Re-running the previous best 40 on 24 folds instead of 12 (`tools/revalidate.py`):

    5f31862f  +0.9293 -> t -0.81   256 trades   RR 3.23
    157a9987  +0.8849 -> t -1.33   178 trades   RR 3.33
    d6060b6a  +0.8785 -> t -1.35   248 trades   RR 3.66
    a6b505e2  +0.8747 -> t -1.89   181 trades   RR 3.72

Of the 15 that completed before the run was cut short, twelve went outright
negative and the other three landed at t = 0.05 / 0.24 / 0.47. None survived.
The edges were specific to the ~9-month window the search had been scoring on —
which is exactly the failure mode the holdout was added to catch, arriving a
step earlier than expected. 555 strategies are now `legacy`.

**Score scale clamped** (`nqq/metrics.py`)
With a t-stat near zero the `raw / max(penalty, 1e-3)` branch produced scores
like -307, far outside the documented (-1, +1) range. The ordering was never
wrong, the scale was: the divisor is floored at 0.25 and the result clamped.
42 stored scores were repaired with `tools/rescore.py`.

**State after the reset:** 62 strategies scored on the new window, best +0.338
(278 trades, RR 1.76, t 2.07, 15/24 folds profitable) against a noise ceiling of
t ~ 2.19 at 62 results. Nothing has cleared the bar yet, which is the honest
position to be in after four hours rather than a leaderboard of +0.9s that meant
nothing.

Two search workers are now running against the same registry (supported: WAL +
atomic `reserve()`), which offsets part of the throughput cost of 24 folds.

---

## 2026-08-17 — tick 4 · sample size, honestly

Prompted by the user asking whether 244 trades is too low. It is, but the reason
is not the count — it is the count relative to the size of the search.

At 500 scored / 915 evaluated genomes, the expected best t-statistic from *pure
noise* is ~2.9–3.1 (Gumbel approximation, `sqrt(2 ln N)` corrected). Measured:

    85b2c3ab (score +0.744, 244 trades)   t = 2.82   below the noise ceiling
    e0700c4c (score +0.831, 121 trades)   t = 2.30   well below it

Both were indistinguishable from the best of 500 coin flips.

**Objective: evidence penalty is now the t-stat, not the trade count**
(`nqq/metrics.py`) `min(1, trades/min_trades)**0.5` ignored the size of the edge,
which is why a 121-trade strategy could outrank a 244-trade one with a better
Sharpe. Now `min(1, t_stat/T_TARGET)` with `T_TARGET = 3.50`, still multiplied by
a (gentler) trade-count term so that a huge edge over 30 trades — a story, not a
strategy — cannot pass on t alone. `t_stat` is stored as a metric.

**Test window widened: `max_folds` 12 → 24**
12 folds covered only ~9 months of the 9 years available. Trade counts roughly
doubled (639 on the first new evaluation). Throughput fell from ~360 to ~97
strategies/hour, which is the right trade: fewer, better-measured hypotheses.

**A real holdout: the last 12 months are amputated** (`nqq/backtest.py`, `config.yaml`)
`walkforward.holdout_months: 12`. `Evaluator.n_searchable` truncates every fold,
so no training window, test window or barrier walk in the search can reach data
after 2024-10-01 (23,389 bars). `tools/holdout.py` is the only caller of
`evaluate(holdout=True)`, and it is never invoked by the loop. The walk-forward
folds stop being out-of-sample once the search selects on them; this is the only
untouched data left, and it gets spent once.

**`tools/revalidate.py`**
A test-window change cannot be rescored arithmetically the way an objective
change can — the trades themselves differ. So every completed strategy was
marked `legacy` (retained for dedupe, removed from the leaderboard *and the
breeding pool*, since a stale score has no business selecting parents) and the
previous best 40 were re-run on the new window.

---

## 2026-08-17 — tick 3 · the elite pool had inbred

Measured at generation 9, over 245 completed strategies. The score-sorted top-25
breeding pool had collapsed onto a single lineage:

    label.tp_mult   19/25 = 3.0        filter.session   23/25 = globex
    label.sl_mult   15/25 = 1.5        trend_align      24/25 = True
    tp/sl geometry  16/25 = 2.0        allow_short      25/25 = True

Crossover was mixing a strategy with itself, and the high-RR lineages had been
squeezed out by slightly-better-scoring 1.8-RR ones. Realised RR tracks the
nominal tp/sl geometry closely (median gap 0.21R), so a pool stuck at geometry
2.0 caps realised RR at roughly 1.8 no matter how long it runs.

**Niched breeding pool** (`nqq/search.py`)
`_elites()` now addresses each strategy by (reward-to-risk band, session) and
gives every niche one seat before any niche gets a second, filling the rest by
score. Rank-weighted selection is unchanged on top, so pressure toward the best
is intact — only the monoculture is gone. Pool now spans 14 distinct niches
instead of being 3/4 one genotype.

**tp_mult ceiling 4.0 → 5.5** (`nqq/genome.py`)
The pool's highest-RR members sat exactly at the old ceiling, so reachable RR
was limited by the search space rather than by the market. Existing fingerprints
are unaffected — canonical hashing snaps to the step grid, not the range.

**Elite pool cached per iteration** (`nqq/search.py`)
`_propose` re-queried SQLite and re-niched on every candidate draw, up to 400
per iteration. It now refreshes once per completed evaluation, which is the only
time the pool can actually change.

---

## 2026-08-17 — tick 2 · reward-to-risk

Standing instruction from the user: **always try to get the RR higher.** The
objective had no reward/risk term at all, so the search was free to breed
high-hit-rate, 1.2-payoff systems.

**Objective: realised reward-to-risk is now scored** (`nqq/metrics.py`)
New metric `payoff_ratio` = avg win R / |avg loss R| — the *realised* RR, which
is well below the nominal tp/sl geometry once timeouts and cost drag are in.
Weights are now expectancy 0.34 / sharpe 0.22 / recovery 0.22 / **payoff 0.22**.
The payoff term is credited only when expectancy is positive: a fat average
winner on a losing system is not an achievement and must not buy score.

**Genome: reward must be at least risk** (`nqq/genome.py`)
`repair()` allowed `tp_mult` down to 0.6x `sl_mult`. A target inside the stop
needs a >50% hit rate to break even *before* the 1.0-point round-trip cost.
Now `tp_mult >= sl_mult`, snapped to the tp grid so fingerprints stay canonical.

**Haircuts moved into `metrics.apply_fold_haircuts()`**
Previously inline in the evaluator, which meant anything rescoring stored
metrics would silently disagree with the live scorer.

**`tools/rescore.py`**
Changing the objective invalidates every score in the DB — and worse, the loop
would keep breeding from elites chosen by the old ruler. This recomputes `score`
from stored metrics (exact, since metrics are objective-independent) and
back-derives `payoff_ratio` for rows written before it existed. All 151
completed strategies were rescored before the searcher was restarted.

**RR is now visible everywhere it is ranked**
`RR` column in the leaderboard and `--status`, `reward/risk` card in the
per-strategy report, `RR x.xx` in the live log line.

Effect on the ranking: the 2.72-payoff strategy climbed from #7 to #6 while the
1.23-payoff one dropped from #4 to #7.

---

## 2026-08-17 — tick 1

**Data loader: delimiter sniffing** (`nqq/data.py`)
`_load_csv` read every file with pandas' comma default, so the tab-separated
MT5-style export in this repo parsed as a single column. The header line is now
sniffed for tab / semicolon / comma / pipe.

**Data loader: tick-volume fallback** (`nqq/data.py`)
The supplied `15m_data.csv` has `Volume` identically zero and the real activity
in `TickVolume`. A zero volume column is worse than a missing one — it silently
zeroes every volume-derived feature instead of failing. When `volume` sums to
zero and a tick-count column exists, tick counts are used and a warning is logged.

**Config: real data wired up** (`config.yaml`)
`data.path` → `15m_data.csv` (206,703 bars, 2016-11-14 → 2025-10-01).
`source_tz` → `Europe/Athens`: the timestamps are broker server time, confirmed
by the volume profile — the busiest 15m bar of the day sits at 16:30 local,
which is 09:30 New York.

**Search: mid-evaluation abort on hopeless trade counts** (`nqq/backtest.py`)
The static pre-gate cannot see the model, so genomes with a too-strict signal
threshold paid for all 12 folds before being rejected on trade count — the
largest single consumer of wasted compute (roughly half of all evaluations in
the first log sample were `only N trades` rejections). `_hopeless()` now aborts
past the halfway fold when 3x the observed trades-per-fold rate still cannot
reach `min_trades`. Deliberately generous, because a rejection is permanent.

**Ops: `run_forever.ps1`**
Console-free crash-restarting supervisor, so the search can run detached
(`run_forever.bat` needs a console for its `timeout` call).
