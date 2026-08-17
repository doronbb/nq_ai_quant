# NQ strategy search

An ML strategy-discovery engine for Nasdaq-100 futures. It generates strategies,
walk-forward tests them, scores them in R-multiples, remembers every one it has
ever tried, and breeds better ones from the winners — on its own, indefinitely,
with nobody watching.

---

## Quick start

```bash
pip install -r requirements.txt
pip install lightgbm            # strongly recommended: ~15x faster than the fallback
```

**1. Verify the engine is correct** (do this first, it takes ~30 seconds):

```bash
python tests/test_core.py
```

35 checks, including the lookahead test. If any fail, stop and fix before trusting a number.

**2. Point it at data.** Edit `config.yaml` → `data.path` to a folder of NQ OHLCV
CSVs. No data yet? Generate synthetic bars to confirm the plumbing works:

```bash
python tools/make_sample_data.py --bars 60000 --out data/sample/NQ_synthetic.csv
# then set data.path: data/sample/  in config.yaml
```

**3. Turn it loose:**

```
Windows :  double-click run_forever.bat
Mac/Linux: ./run_forever.sh          (or: nohup ./run_forever.sh & )
```

**4. Watch it.** Open `results/leaderboard.html` in a browser and refresh whenever
you like. It rewrites every 10 strategies.

```bash
python run_search.py --status       # same thing in the terminal
```

Ctrl+C once stops it cleanly. Start it again any time — it resumes exactly where
it stopped.

---

## What "never tries the same thing twice" actually means

Every strategy is a **genome**: a dict of ~24 genes covering barriers, features,
model hyperparameters, signal thresholds and entry filters. The genome is reduced
to a canonical form (keys sorted, floats snapped to the search grid, feature lists
sorted) and hashed into a 20-character **fingerprint**.

`results/strategies.db` (SQLite) stores every fingerprint ever *considered*, and
the row is written **before** any compute is spent. Consequences:

- A duplicate is discarded in microseconds, before the model is ever fit.
- Killing the process mid-evaluation cannot cause a repeat — the reservation persists.
- Failures and hard rejections are stored too, so a genome that produced 0 trades
  or threw an exception is never retried either.
- `learning_rate=0.05` and `learning_rate=0.0500001` are the *same* experiment.
  Without grid-snapping, a float-mutating search would happily re-test the same
  strategy forever under new hashes.

Delete `results/strategies.db` to start the search over from nothing. Keep it and
the search compounds across weeks.

Verify it yourself any time:

```bash
sqlite3 results/strategies.db \
  "select count(*), count(distinct fingerprint), count(distinct genome) from strategies;"
# the three numbers are always identical
```

---

## How it improves, based on R

**Phase 1 — exploration.** The first `search.warmup` (default 60) strategies are
uniform random draws. This builds an unbiased map of the space instead of
committing early to whatever the first lucky genome looked like.

**Phase 2 — evolution.** After warmup, each candidate is drawn from:

| source | share | what it does |
|---|---|---|
| mutation of an elite | ~55% | point-mutates genes; numeric genes take a local step |
| crossover of two elites | ~25% | uniform gene mixing, biased toward the better parent |
| fresh random | ~20% | keeps the search from collapsing onto one lineage |

"Elite" means top-25 by composite score, sampled **rank-weighted** so rank 1 is
about 25x likelier to breed than rank 25 — pressure toward the best without
inbreeding on a single winner.

When the neighbourhood around the elites is exhausted (many consecutive
duplicates), the loop automatically raises the mutation rate and the random
share. That is how it escapes a saturated region instead of spinning.

### The objective

Composite of the three metrics you'd otherwise have to trade off by hand:

```
score = [ 0.40·tanh(expectancy / 0.20)
        + 0.30·tanh(sharpe     / 1.50)
        + 0.30·tanh(recovery   / 3.00) ] × √(min(1, trades / min_trades))
```

`tanh` bounds each component, so no single metric can be gamed to dominate. Then
two haircuts fight curve-fitting:

- **consistency**: multiplied by `0.4 + 0.6 × (fraction of folds that were profitable)`
- **concentration**: penalised when >70% of total R came from one fold

A strategy that made all its money in one lucky quarter scores near zero, which
is the point.

---

## Anti-overfitting measures

These are the parts worth arguing with, so they're all explicit:

**Walk-forward only.** Folds are strictly chronological, rolling (or anchored).
No shuffling, ever. Only the most recent `max_folds` are kept, because 2015 NQ
regime behaviour is weak evidence about today's.

**Embargo.** Between the end of a training window and the start of its test
window, `max_bars` bars are dropped. A triple-barrier label at bar *t* peeks
forward to *t+max_bars*, so without the embargo the last training labels overlap
the test period. This is the leak that quietly inflates most published
triple-barrier backtests.

**Causal features.** `tests/test_core.py` truncates the data at bar *k*, rebuilds
every feature from scratch, and asserts each value at bars ≤ *k* is bit-identical
to the full-history version. Any future leakage fails the test loudly.

**Realistic fills.** Signal from bar *t*'s close → fill at bar *t+1*'s **open**.
When both barriers fall inside one bar, the **stop** is assumed to hit first —
you cannot know the intrabar path, so take the pessimistic branch.

**Costs from the start.** `costs.round_trip_points` (default 1.0 NQ point ≈ $20)
is subtracted from every trade's numerator. A stop-out is therefore slightly
worse than −1R, as it is in reality. A strategy that only works at zero cost
isn't a strategy.

**One position at a time.** A new signal is ignored until the open trade's exit
bar has passed, so overlapping trades can't inflate the trade count.

**Hard gates.** Fewer than `min_trades` out-of-sample trades or fewer than
`min_folds` usable folds → permanently rejected. 20 trades of +0.8R is noise.

---

## R-multiple accounting

```
R = (exit − entry) / (entry − stop),  sign-adjusted for direction
```

Barriers are ATR multiples set at signal time: `stop = sl_mult × ATR`,
`target = tp_mult × ATR`, both genes in the search. Every reported number is in
R and never in dollars, so nothing depends on account size.

Per strategy you get: trade count, win rate, avg R, avg win R, avg loss R,
expectancy, total R, max drawdown in R, recovery factor, Sharpe on the R series
(annualised by realised trade frequency), profit factor, longest losing streak,
plus per-fold breakdowns.

---

## Output layout

```
results/
  leaderboard.html               live ranking — open in a browser, refresh freely
  strategies.db                  every strategy ever tried (the dedupe memory)
  search.log                     full history
  cache/                         parsed + resampled bars
  strategies/
    <fingerprint>.html           equity curve in R, fold table, feature importance,
                                 trade log preview, full genome
    <fingerprint>_trades.csv     entry/exit time, direction, entry, stop, target,
                                 exit, risk points, bars held, confidence, R
    <fingerprint>_genome.json    exact config to reproduce it
```

Reports use hand-rolled inline SVG — no matplotlib, no server. The HTML opens
anywhere, including from a USB stick.

---

## Data sources

| source | reliability | history | key | notes |
|---|---|---|---|---|
| `csv` | highest | whatever you have | no | **default.** Works offline. Named or headerless columns. |
| `databento` | highest | full | yes | `GLBX.MDP3` / `NQ.c.0` continuous front month. Best if you want it self-updating. |
| `yfinance` | low | ~60 days | no | Gappy and thin. Bootstrap only — the leaderboard shows a warning banner when it's active. |

CSV is the default deliberately: it can't rate-limit you, can't change its schema
overnight, and can't quietly stop working at 3am while the searcher runs on.
Databento is the upgrade when you want the system to keep itself current.

---

## Tuning throughput

Roughly, on 8k bars with the numpy fallback: ~90 strategies/hour. With LightGBM
installed: ~15x that.

- **Install LightGBM.** Single biggest win.
- `walkforward.max_folds` — fewer folds, faster, noisier. 8–12 is a good band.
- `search.barrier_cache` — barrier walks are memoised across genomes sharing
  `(atr_period, tp, sl, max_bars)`. Raise it if you have RAM; the hit rate climbs
  sharply once evolution starts.
- Coarser `timeframe` (15min vs 1min) is dramatically faster and usually less
  noisy.
- Run two copies against the same `results/` folder — SQLite is in WAL mode and
  `reserve()` is atomic, so workers can't collide on the same genome.

---

## Reading the results honestly

The engine is a hypothesis generator, not an oracle. It will find things that
look good on any dataset, including pure noise — that is what a search does.

Before believing anything on the leaderboard:

1. **Check fold consistency.** `folds+` on the leaderboard. 7/8 profitable folds
   is a signal; 3/8 with one huge fold is a curve fit that slipped past the
   haircut.
2. **Check the trade count.** 400 trades beats 120 trades at the same expectancy.
3. **Check the feature importance.** If the top features are unstable across the
   nearby strategies in the ranking, the "edge" is probably noise.
4. **Raise `costs.round_trip_points` and re-run it.** An edge that dies at 2
   points was never an edge.
5. **Hold out data the search never saw.** Set `data.end` to a year ago, let the
   search run, then test the winner on the excluded period. This is the only
   test that really counts, because the search itself has effectively used up
   your other out-of-sample data by selecting on it.

Point 5 is the one people skip. With thousands of strategies tested, the
best-scoring one is selected partly on luck — the walk-forward folds stop being
out-of-sample once you've mined them. A truly untouched holdout is the only
honest verdict.

---

## Layout

```
config.yaml            everything you'd want to change
run_search.py          CLI: --iters --status --report --reset --no-cache
run_forever.bat/.sh    crash-restarting wrappers — "set and forget"
nqq/
  data.py              adapters, resampling, session tagging, caching
  features.py          70 causal features in 7 switchable groups
  labeling.py          triple barrier + memoised barrier walks
  genome.py            search space, canonical hashing, mutation, crossover
  registry.py          SQLite dedupe memory
  model.py             lightgbm → sklearn → built-in numpy GBDT
  backtest.py          walk-forward evaluation, R accounting
  metrics.py           R metrics + the composite objective
  search.py            the autonomous loop
  report.py            leaderboard + per-strategy HTML
tests/test_core.py     35 checks, lookahead test included
tools/make_sample_data.py   synthetic bars for pipeline verification
```

---

## Not included, on purpose

No live trading, no broker connection, no order routing. This finds and measures
hypotheses. Wiring a discovered strategy to real money is a separate decision
with separate risks, and it shouldn't happen by accident because a script was
left running.

Nothing here is financial advice, and a backtest is not a prediction.
