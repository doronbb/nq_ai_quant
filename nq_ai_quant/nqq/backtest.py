"""
Walk-forward evaluation of a genome, in R-multiples.

Time-order discipline
---------------------
* Folds are strictly chronological. There is no shuffling anywhere.
* Between the end of a training window and the start of its test window we
  EMBARGO `max_bars` bars, because a training label at bar t peeks forward up
  to t+max_bars. Without the embargo the last training labels would overlap the
  test period — the classic subtle leak in triple-barrier setups.
* Features at bar t use bars <= t; the fill is at open[t+1]. So even a signal
  generated on the final tick of bar t is tradeable.
* Only one position at a time: a new signal is ignored until the open trade's
  barrier exit has passed.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .features import adx, atr, build_features, select_columns
from .labeling import BarrierCache
from .model import make_model, available_backend

log = logging.getLogger("nqq.backtest")


class Evaluator:
    """
    Holds everything that is expensive and genome-independent (bars, the full
    feature matrix, ATR series, barrier cache) so evaluating one more genome is
    cheap. Build it once, call `evaluate()` forever.
    """

    def __init__(self, bars: pd.DataFrame, cfg: dict):
        self.bars = bars
        self.cfg = cfg
        wf = cfg["walkforward"]
        self.train_bars = int(wf["train_bars"])
        self.test_bars = int(wf["test_bars"])
        self.max_folds = int(wf.get("max_folds", 0))
        self.anchored = bool(wf.get("anchored", False))
        self.holdout_months = int(wf.get("holdout_months", 0))
        self.min_trades = int(cfg["search"]["min_trades"])
        self.screen_z = float(cfg["search"].get("screen_z", 2.0))
        self.cost_points = float(cfg["costs"]["round_trip_points"])
        self.backend = cfg["model"].get("backend") or available_backend()

        log.info("building feature matrix (%d bars)…", len(bars))
        self.feats, self.groups = build_features(bars)
        self._atr: dict[int, pd.Series] = {}
        self._filter_cache: dict[tuple, np.ndarray] = {}
        self.barriers = BarrierCache(maxsize=int(cfg["search"].get("barrier_cache", 24)))

        # Filter inputs, precomputed once.
        adx14, _, _ = adx(bars, 14)
        self.adx14 = adx14.to_numpy(np.float64)
        self.atr_pct = (self.atr_for(14) / bars["close"] * 100).to_numpy(np.float64)
        self.is_rth = bars["_is_rth"].to_numpy(np.int8)
        self.minute_of_day = bars["_minute_of_day"].to_numpy(np.int32)
        self.session_id = bars.groupby("_session_date", sort=False).ngroup().to_numpy(np.int64)
        self.ema200_dist = self.feats["ema200_dist"].to_numpy(np.float64)
        self.index = bars.index
        self.open = bars["open"].to_numpy(np.float64)

        bpy = self._bars_per_year()
        self.bars_per_year = bpy

        # Amputate the holdout. `n_searchable` is the only length the search is
        # ever allowed to see; everything at or past it is frozen.
        self.n_searchable = len(bars)
        if self.holdout_months > 0:
            cut = self.index[-1] - pd.DateOffset(months=self.holdout_months)
            self.n_searchable = int(self.index.searchsorted(cut))
            log.info("holdout: last %d months frozen (%d bars from %s) — "
                     "the search cannot see them",
                     self.holdout_months, len(bars) - self.n_searchable,
                     self.index[self.n_searchable].date())

        log.info("features: %d cols | backend: %s | ~%.0f bars/year | %d searchable bars",
                 self.feats.shape[1], self.backend, bpy, self.n_searchable)

    # -- helpers -----------------------------------------------------------

    def atr_for(self, period: int) -> pd.Series:
        if period not in self._atr:
            self._atr[period] = atr(self.bars, period)
        return self._atr[period]

    def _bars_per_year(self) -> float:
        span = (self.index[-1] - self.index[0]).total_seconds() / 86400.0
        return len(self.bars) / max(span, 1e-9) * 365.25

    def _folds(self, max_bars: int):
        """Yield (train_slice, test_slice) index arrays, chronological, embargoed."""
        n = self.n_searchable
        embargo = int(max_bars) + 1
        start = self.train_bars + embargo
        folds = []
        pos = start
        while pos + self.test_bars <= n:
            tr_lo = 0 if self.anchored else max(0, pos - embargo - self.train_bars)
            tr_hi = pos - embargo
            if tr_hi - tr_lo >= max(200, self.train_bars // 4):
                folds.append((np.arange(tr_lo, tr_hi), np.arange(pos, pos + self.test_bars)))
            pos += self.test_bars
        if self.max_folds and len(folds) > self.max_folds:
            folds = folds[-self.max_folds:]        # most recent regimes matter most
        return folds

    def _holdout_folds(self, max_bars: int):
        """
        One fold: train on the most recent searchable window, test on the frozen
        period. This is the only evaluation allowed to touch the holdout, and it
        is never called by the search loop — only by tools/holdout.py, once you
        have already decided which strategy you believe in.
        """
        n = len(self.bars)
        if self.n_searchable >= n:
            return []
        embargo = int(max_bars) + 1
        tr_hi = self.n_searchable - embargo
        tr_lo = 0 if self.anchored else max(0, tr_hi - self.train_bars)
        if tr_hi - tr_lo < max(200, self.train_bars // 4):
            return []
        return [(np.arange(tr_lo, tr_hi), np.arange(self.n_searchable, n))]

    def _static_filter_mask(self, g: dict) -> np.ndarray:
        """
        Direction-independent part of the entry filters, vectorised over all bars.
        Used by the pre-gate; `_passes_filters` remains the authority per trade.
        """
        key = (g["filter.session"], g["filter.adx_min"],
               round(g["filter.atr_pct_min"], 4), round(g["filter.atr_pct_max"], 4))
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached

        sess, rth, mod = g["filter.session"], self.is_rth, self.minute_of_day
        if sess == "rth":
            m = rth == 1
        elif sess == "globex":
            m = rth == 0
        elif sess == "rth_no_open":
            m = (rth == 1) & (mod >= 600)
        elif sess == "open_drive":
            m = (mod >= 570) & (mod < 630)
        else:
            m = np.ones(len(rth), bool)

        if g["filter.adx_min"]:
            m &= np.nan_to_num(self.adx14, nan=-1) >= g["filter.adx_min"]
        ap = self.atr_pct
        m &= np.isfinite(ap) & (ap >= g["filter.atr_pct_min"]) & (ap <= g["filter.atr_pct_max"])

        self._filter_cache[key] = m
        if len(self._filter_cache) > 64:
            self._filter_cache.pop(next(iter(self._filter_cache)))
        return m

    def _passes_filters(self, g: dict, i: int, direction: int) -> bool:
        sess = g["filter.session"]
        rth = self.is_rth[i]
        mod = self.minute_of_day[i]
        if sess == "rth" and not rth:
            return False
        if sess == "globex" and rth:
            return False
        if sess == "rth_no_open" and (not rth or mod < 600):     # skip first 30m
            return False
        if sess == "open_drive" and not (570 <= mod < 630):
            return False
        if g["filter.adx_min"] and not (self.adx14[i] >= g["filter.adx_min"]):
            return False
        ap = self.atr_pct[i]
        if not np.isfinite(ap) or ap < g["filter.atr_pct_min"] or ap > g["filter.atr_pct_max"]:
            return False
        if g["filter.trend_align"]:
            d = self.ema200_dist[i]
            if not np.isfinite(d):
                return False
            if direction > 0 and d < 0:
                return False
            if direction < 0 and d > 0:
                return False
        return True

    # -- main --------------------------------------------------------------

    def evaluate(self, g: dict, collect_trades: bool = True,
                 holdout: bool = False) -> dict:
        """
        Returns dict with: metrics, fold_metrics, consistency, trades (DataFrame),
        importance (dict), n_folds, elapsed.
        Raises nothing on 'strategy is bad' — that shows up as a low score.
        """
        t0 = time.time()

        atr_s = self.atr_for(int(g["label.atr_period"]))
        bar = self.barriers.get(
            self.bars, atr_s, float(g["label.tp_mult"]), float(g["label.sl_mult"]),
            int(g["label.max_bars"]), self.cost_points, int(g["label.atr_period"]),
        )

        cols = select_columns(self.groups, g["feat.groups"])
        if len(cols) < 3:
            return self._null_result("too few feature columns", time.time() - t0)

        X_all = self.feats[cols].to_numpy(np.float64)
        y_all = bar["label"].to_numpy(np.int8)
        valid = bar["valid"].to_numpy(bool) & np.isfinite(X_all).any(axis=1)
        # A row is trainable only if enough of its features exist.
        enough_feats = np.isfinite(X_all).mean(axis=1) > 0.8
        trainable = valid & enough_feats

        long_R = bar["long_R"].to_numpy(np.float64)
        short_R = bar["short_R"].to_numpy(np.float64)
        long_exit = bar["long_exit"].to_numpy(np.int64)
        short_exit = bar["short_exit"].to_numpy(np.int64)
        entry_px = bar["entry_price"].to_numpy(np.float64)
        l_stop = bar["long_stop"].to_numpy(np.float64)
        s_stop = bar["short_stop"].to_numpy(np.float64)
        l_tgt = bar["long_target"].to_numpy(np.float64)
        s_tgt = bar["short_target"].to_numpy(np.float64)

        max_bars = int(g["label.max_bars"])
        folds = self._holdout_folds(max_bars) if holdout else self._folds(max_bars)
        if not folds:
            return self._null_result(
                "no holdout period configured" if holdout
                else "not enough data for one walk-forward fold", time.time() - t0)

        # Cheap pre-gate. The static filters (session / ADX / ATR band) do not
        # depend on the model, so we can count admissible test bars before
        # spending 20+ seconds on a fit that could only ever produce 0 trades.
        # Budget: even a perfect model can only trade a fraction of admitted
        # bars, since a position blocks the next `max_bars` bars.
        test_all = np.concatenate([te for _, te in folds])
        admitted = self._static_filter_mask(g)[test_all].sum()
        capacity = admitted / max(int(g["label.max_bars"]) * 0.35, 1.0)
        # The holdout is one short fold by construction — the min_trades budget
        # is meaningless there, and rejecting on it would defeat the whole point.
        if not holdout and capacity < self.min_trades:
            return self._null_result(
                f"filters admit only {int(admitted)} test bars — at max_bars="
                f"{g['label.max_bars']} that caps out near {int(capacity)} trades "
                f"(need {self.min_trades})", time.time() - t0)

        classes = [-1, 0, 1]
        mp = {
            "n_estimators": int(g["model.n_estimators"]),
            "learning_rate": float(g["model.learning_rate"]),
            "num_leaves": int(g["model.num_leaves"]),
            "max_depth": int(g["model.max_depth"]),
            "min_child_samples": int(g["model.min_child_samples"]),
            "subsample": float(g["model.subsample"]),
            "colsample": float(g["model.colsample"]),
            "reg_lambda": float(g["model.reg_lambda"]),
            "seed": int(self.cfg["search"].get("seed", 7)),
            "threads": int(self.cfg["model"].get("threads", 0)),
        }

        trades: list[dict] = []
        scored_folds: list[tuple[int, dict]] = []
        importance = np.zeros(len(cols))
        max_per_day = int(g["filter.max_trades_per_day"])
        done = 0

        def run_folds(fold_ids) -> str | None:
            """Fit and simulate the given folds. Returns an abort reason or None."""
            nonlocal done
            for fi in fold_ids:
                tr_idx, te_idx = folds[fi]
                tr = tr_idx[trainable[tr_idx]]
                if len(tr) < 300 or len(np.unique(y_all[tr])) < 2:
                    continue

                Xtr = np.nan_to_num(X_all[tr], nan=0.0, posinf=0.0, neginf=0.0)
                ytr = y_all[tr]
                present = [c for c in classes if (ytr == c).sum() >= 10]
                if len(present) < 2:
                    continue

                try:
                    model = make_model(mp, self.backend).fit(Xtr, ytr, present)
                except Exception as e:                  # a bad hyperparam combo
                    log.debug("fold %d fit failed: %s", fi, e)
                    continue

                te = te_idx[valid[te_idx]]
                if len(te) == 0:
                    continue
                Xte = np.nan_to_num(X_all[te], nan=0.0, posinf=0.0, neginf=0.0)
                proba = model.predict_proba(Xte)

                ci = {c: k for k, c in enumerate(present)}
                p_long = proba[:, ci[1]] if 1 in ci else np.zeros(len(te))
                p_short = proba[:, ci[-1]] if -1 in ci else np.zeros(len(te))

                try:
                    imp = np.asarray(model.feature_importance(), dtype=float)
                    if imp.shape == importance.shape:
                        tot = imp.sum()
                        if tot > 0:
                            importance += imp / tot
                except Exception:
                    pass

                fold_trades = self._simulate(
                    g, te, p_long, p_short, long_R, short_R, long_exit, short_exit,
                    entry_px, l_stop, s_stop, l_tgt, s_tgt, max_per_day, fi,
                )
                trades.extend(fold_trades)
                scored_folds.append((fi, M.compute(
                    pd.DataFrame(fold_trades) if fold_trades else pd.DataFrame(),
                    self.bars_per_year, self.min_trades)))
                done += 1

                if not holdout:
                    abort = self._hopeless(len(trades), done, len(folds))
                    if abort:
                        return abort
            return None

        # ---- two-stage evaluation ----------------------------------------
        # Stage 1 fits a spread-out subset of folds; a genome that is clearly
        # losing across the whole period is dropped before the remaining folds
        # are ever fit. Stage 2 evaluates only the folds stage 1 did not, so a
        # genome that survives pays no extra cost — the screen is free on
        # winners and saves ~3/4 of the work on losers.
        screen_n = int(self.cfg["search"].get("screen_folds", 0))
        use_screen = (not holdout) and screen_n >= 3 and len(folds) >= screen_n * 2
        if use_screen:
            step = len(folds) / screen_n
            screen_ids = sorted({min(len(folds) - 1, int(i * step)) for i in range(screen_n)})
        else:
            screen_ids = list(range(len(folds)))

        abort = run_folds(screen_ids)
        if abort:
            return self._null_result(abort, time.time() - t0)

        if use_screen:
            verdict = self._screen_verdict(trades, len(screen_ids), len(folds))
            if verdict:
                return self._null_result(verdict, time.time() - t0)
            rest = [i for i in range(len(folds)) if i not in set(screen_ids)]
            abort = run_folds(rest)
            if abort:
                return self._null_result(abort, time.time() - t0)

        # The screen evaluates folds out of order, so both the fold table and the
        # trade sequence have to be put back into time order. The trades matter
        # most: drawdown and the equity curve are path-dependent, so an
        # out-of-order list silently produces a different max_dd_R — and through
        # the recovery factor, a different score — for the same strategy.
        fold_metrics = [m for _, m in sorted(scored_folds, key=lambda x: x[0])]
        trades.sort(key=lambda t: (t["fold"], t["signal_time"]))

        tdf = pd.DataFrame(trades)
        met = M.compute(tdf, self.bars_per_year, self.min_trades)
        cons = M.fold_consistency(fold_metrics)
        met.update({f"fold_{k}": v for k, v in cons.items()})

        met["score"] = M.apply_fold_haircuts(met["score"], cons)

        imp_map = {}
        if importance.sum() > 0:
            norm = importance / importance.sum()
            imp_map = {c: round(float(v), 5) for c, v in
                       sorted(zip(cols, norm), key=lambda x: -x[1])}

        return {
            "metrics": met,
            "fold_metrics": fold_metrics,
            "consistency": cons,
            "trades": tdf if collect_trades else pd.DataFrame(),
            "importance": imp_map,
            "n_folds": len(fold_metrics),
            "elapsed": time.time() - t0,
        }

    def _screen_verdict(self, trades: list, n_screened: int, n_total: int) -> str | None:
        """
        Stage-1 verdict. Rejects only what a full evaluation could not rescue.

        Two ways to fail: the trade pace cannot reach `min_trades`, or the edge
        is negative BEYOND SAMPLING ERROR across folds spanning the whole period.

        The error bar is the whole point. A quarter of the folds might hold only
        40 trades, and per-trade R has a standard deviation around 1.4, so the
        standard error on 40 trades is ~0.22R. Comparing a point estimate to a
        fixed threshold there discards good strategies wholesale: the first
        version of this screen rejected a +0.233R / 170-trade strategy sitting
        at rank 2 on the leaderboard, because its 40 screen trades happened to
        read -0.082R. Well inside noise.

        So the test is on the upper confidence bound: reject only when even a
        generous `screen_z` standard errors above the observed mean is still
        losing. That makes rejection mean "cannot plausibly be profitable",
        not "is not profitable in this small sample" — which matters because a
        rejection is written to the registry permanently and the genome is
        never reconsidered.

        Set search.screen_folds to 0 to turn the screen off entirely.
        """
        if n_screened >= n_total:
            return None
        n = len(trades)
        projected = n / max(n_screened, 1) * n_total * 3.0
        if projected < self.min_trades:
            return (f"screened out: {n} trades on {n_screened}/{n_total} spread folds "
                    f"cannot reach {self.min_trades} (early abort)")
        if n < 60:
            return None                      # too thin to judge the edge at all
        R = np.array([t["R"] for t in trades], dtype=np.float64)
        R = R[np.isfinite(R)]
        if len(R) < 60:
            return None
        mean = float(R.mean())
        se = float(R.std(ddof=1)) / np.sqrt(len(R))
        upper = mean + self.screen_z * se
        if upper < 0.0:
            return (f"screened out: {mean:+.3f}R over {n} trades on {n_screened}/"
                    f"{n_total} spread folds — losing even {self.screen_z:.0f} SE "
                    f"high ({upper:+.3f}R)")
        return None

    def _hopeless(self, n_trades: int, folds_done: int, folds_total: int) -> str | None:
        """
        Mid-evaluation abort for genomes that cannot reach `min_trades`.

        The static pre-gate catches filters that admit too few bars, but it cannot
        see the model: a signal threshold of 0.68 may admit thousands of bars and
        still fire twice. Those genomes currently pay for every fold before being
        rejected on trade count, and they are the single largest consumer of
        wasted compute in the search.

        Folds are equal-length, so the observed trades-per-fold rate is a fair
        estimator of the remaining folds. The bound is deliberately generous —
        3x the observed rate — and only applies past the halfway mark with at
        least 3 folds of evidence, because a rejection is written to the registry
        permanently and a genome killed here is never reconsidered.
        """
        if folds_done < 3 or folds_done < folds_total / 2 or folds_done >= folds_total:
            return None
        optimistic = n_trades / folds_done * folds_total * 3.0
        if optimistic >= self.min_trades:
            return None
        return (f"only {n_trades} trades after {folds_done}/{folds_total} folds — "
                f"cannot reach {self.min_trades} (early abort)")

    # -- trade simulation --------------------------------------------------

    def _simulate(self, g, te, p_long, p_short, long_R, short_R, long_exit, short_exit,
                  entry_px, l_stop, s_stop, l_tgt, s_tgt, max_per_day, fold_i):
        lt = float(g["signal.long_thresh"])
        st = float(g["signal.short_thresh"])
        margin = float(g["signal.edge_margin"])
        allow_l = bool(g["signal.allow_long"])
        allow_s = bool(g["signal.allow_short"])

        out = []
        busy_until = -1               # bar index at which the open trade exits
        day_count: dict[int, int] = {}

        for k, i in enumerate(te):
            if i <= busy_until:
                continue
            sid = int(self.session_id[i])
            if day_count.get(sid, 0) >= max_per_day:
                continue

            pl, ps = p_long[k], p_short[k]
            direction = 0
            if allow_l and pl >= lt and (pl - ps) >= margin:
                direction = 1
            elif allow_s and ps >= st and (ps - pl) >= margin:
                direction = -1
            if direction == 0:
                continue
            if not self._passes_filters(g, i, direction):
                continue

            if direction > 0:
                R, ex, stop, tgt, conf = long_R[i], long_exit[i], l_stop[i], l_tgt[i], pl
            else:
                R, ex, stop, tgt, conf = short_R[i], short_exit[i], s_stop[i], s_tgt[i], ps
            if not np.isfinite(R) or ex < 0:
                continue

            entry = entry_px[i]
            risk = abs(entry - stop)
            exit_price = entry + R * risk * direction

            out.append({
                "fold": fold_i,
                "signal_time": self.index[i],
                "entry_time": self.index[i + 1] if i + 1 < len(self.index) else self.index[i],
                "exit_time": self.index[int(ex)],
                "direction": "long" if direction > 0 else "short",
                "entry_price": round(float(entry), 2),
                "stop_price": round(float(stop), 2),
                "target_price": round(float(tgt), 2),
                "exit_price": round(float(exit_price), 2),
                "risk_points": round(float(risk), 2),
                "bars_held": int(ex - i),
                "confidence": round(float(conf), 4),
                "R": round(float(R), 4),
            })
            busy_until = int(ex)
            day_count[sid] = day_count.get(sid, 0) + 1

        return out

    @staticmethod
    def _null_result(reason: str, elapsed: float) -> dict:
        m = M._empty()
        m["reject_reason"] = reason
        return {"metrics": m, "fold_metrics": [], "consistency": M.fold_consistency([]),
                "trades": pd.DataFrame(), "importance": {}, "n_folds": 0, "elapsed": elapsed}
