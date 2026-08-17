"""
The autonomous search loop.

How it improves over time
-------------------------
Phase 1 (exploration)  — until `warmup` strategies have been scored, sample the
                         space uniformly at random. Builds an unbiased map.
Phase 2 (evolution)    — thereafter, each new candidate comes from:
                            mutation of an elite      (p_mutate)
                            crossover of two elites   (p_cross)
                            fresh random              (p_random)
                         "Elite" = top-K by composite R score, sampled with
                         rank-weighted probability so the search does not
                         collapse onto one lineage.

Every candidate is fingerprinted and checked against the registry BEFORE any
compute is spent. If it has been seen, it is discarded and another is drawn.
When the local neighbourhood is exhausted (too many consecutive duplicates) the
loop temporarily raises the mutation rate and the random share, which is how it
escapes a saturated region instead of spinning.

Ctrl+C is handled: the loop finishes the current genome, writes the report and
exits cleanly. Restarting resumes from the database with zero repeated work.
"""
from __future__ import annotations

import logging
import random
import signal
import time

from . import report as R
from .backtest import Evaluator
from .genome import crossover, describe, fingerprint, mutate, random_genome
from .registry import Registry

log = logging.getLogger("nqq.search")

# Reward-to-risk band edges (tp_mult / sl_mult) used to keep the breeding pool
# spread across RR regimes instead of collapsing onto whatever won first.
_GEOM_BANDS = (1.5, 2.0, 3.0, 4.0)


class SearchLoop:
    def __init__(self, cfg: dict, bars, registry: Registry):
        self.cfg = cfg
        s = cfg["search"]
        self.reg = registry
        self.ev = Evaluator(bars, cfg)
        self.rng = random.Random(s.get("seed", 7))

        self.warmup = int(s.get("warmup", 60))
        self.elite_k = int(s.get("elite_k", 25))
        self.p_mutate = float(s.get("p_mutate", 0.55))
        self.p_cross = float(s.get("p_crossover", 0.25))
        self.min_trades = int(s.get("min_trades", 100))
        self.min_folds = int(s.get("min_folds", 3))
        self.report_every = int(s.get("report_every", 10))
        self.max_dup_tries = int(s.get("max_duplicate_tries", 400))
        self.save_top = int(s.get("save_full_report_top_n", 10))

        self.stop = False
        self._install_signals()

        self.evaluated = 0
        self.duplicates_skipped = 0
        self.t_start = time.time()
        self._elite_cache: tuple[list[dict], int] | None = None

    # -- graceful shutdown -------------------------------------------------

    def _install_signals(self):
        def handler(signum, frame):
            if self.stop:
                log.warning("second interrupt — exiting now")
                raise KeyboardInterrupt
            log.warning("interrupt received — finishing current strategy, then stopping")
            self.stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, AttributeError, OSError):
                pass       # not the main thread / unsupported platform

    # -- candidate generation ---------------------------------------------

    @staticmethod
    def _niche(g: dict) -> tuple:
        """
        Coarse behavioural address of a strategy: its reward-to-risk band and
        the session it trades. Two genomes in the same cell are near-substitutes
        as breeding stock however differently their genes are spelled.
        """
        geom = g["label.tp_mult"] / max(g["label.sl_mult"], 1e-9)
        band = sum(geom >= edge for edge in _GEOM_BANDS)
        return (band, g["filter.session"])

    def _elites(self) -> list[dict]:
        """
        Breeding pool, one seat per niche before any niche gets a second.

        A plain score-sorted top-K collapses: by generation 8 the pool held 19
        copies of tp3.0/sl1.5 and 23 of 25 traded globex, so crossover was mixing
        a strategy with itself and the high-RR lineages had been squeezed out
        entirely by slightly-better-scoring 1.8-RR ones. Reserving a seat for the
        best member of each reward-to-risk band keeps those lineages alive and
        breeding. Rank-weighted selection still applies on top, so the pressure
        toward the best is unchanged — only the monoculture is gone.
        """
        pool = self.reg.top(self.elite_k * 4)          # already score-descending
        if len(pool) <= self.elite_k:
            return pool

        best_per_niche: dict[tuple, dict] = {}
        runners_up: list[dict] = []
        for r in pool:
            key = self._niche(r["genome"])
            if key in best_per_niche:
                runners_up.append(r)
            else:
                best_per_niche[key] = r

        chosen = list(best_per_niche.values())[:self.elite_k]
        for r in runners_up:
            if len(chosen) >= self.elite_k:
                break
            chosen.append(r)
        chosen.sort(key=lambda r: -r["score"])
        return chosen

    def _pick_elite(self, elites: list[dict]) -> dict:
        """Rank-weighted pick: rank 1 is ~elite_k times likelier than rank K."""
        n = len(elites)
        weights = [n - i for i in range(n)]
        return self.rng.choices(elites, weights=weights, k=1)[0]

    def _elites_cached(self) -> tuple[list[dict], int]:
        """
        The elite pool changes at most once per completed evaluation, but
        `_next_unseen` can draw up to `max_duplicate_tries` (400) candidates.
        Querying SQLite and re-niching on every draw cost more than generating
        the genomes did. Refreshed once per iteration by `run()`.
        """
        if self._elite_cache is None:
            self._elite_cache = (self._elites(), self.reg.stats()["ok"] or 0)
        return self._elite_cache

    def _propose(self, desperation: float) -> tuple[dict, str, int, list[str]]:
        elites, n_done = self._elites_cached()

        if n_done < self.warmup or not elites:
            return random_genome(self.rng), "random", 0, []

        p_random = min(0.9, 0.20 + desperation)
        r = self.rng.random()
        if r < p_random:
            return random_genome(self.rng), "random", 0, []

        rate = min(0.65, 0.20 + 0.5 * desperation)
        if r < p_random + self.p_cross and len(elites) >= 2:
            a = self._pick_elite(elites)
            b = self._pick_elite(elites)
            if a["fingerprint"] == b["fingerprint"]:
                b = self.rng.choice(elites)
            child = crossover(a["genome"], b["genome"], self.rng)
            if self.rng.random() < 0.5:
                child = mutate(child, self.rng, rate * 0.5)
            gen = max(a["generation"], b["generation"]) + 1
            return child, "crossover", gen, [a["fingerprint"], b["fingerprint"]]

        p = self._pick_elite(elites)
        return (mutate(p["genome"], self.rng, rate), "mutation",
                p["generation"] + 1, [p["fingerprint"]])

    def _next_unseen(self):
        """Draw until we find a genome the registry has never claimed."""
        tries = 0
        while tries < self.max_dup_tries:
            desperation = tries / self.max_dup_tries
            g, origin, gen, parents = self._propose(desperation)
            fp = fingerprint(g)
            tries += 1
            if self.reg.seen(fp):
                self.duplicates_skipped += 1
                continue
            if self.reg.reserve(fp, g, origin, gen, parents):
                return g, fp, origin, gen, parents
            self.duplicates_skipped += 1
        return None

    # -- main loop ---------------------------------------------------------

    def run(self, max_iters: int = 0):
        self.reg.cleanup_stale_pending()
        log.info("registry: %s", self.reg.stats())
        it = 0
        while not self.stop and (max_iters == 0 or it < max_iters):
            it += 1
            self._elite_cache = None          # one refresh per evaluation
            drawn = self._next_unseen()
            if drawn is None:
                log.warning(
                    "search space looks saturated near the current elites "
                    "(%d consecutive duplicates). Widening: raising random share.",
                    self.max_dup_tries)
                self.p_mutate = min(0.9, self.p_mutate + 0.05)
                time.sleep(2)
                continue

            g, fp, origin, gen, parents = drawn
            t0 = time.time()
            try:
                res = self.ev.evaluate(g, collect_trades=True)
            except KeyboardInterrupt:
                self.reg.fail(fp, "interrupted")
                raise
            except Exception as e:
                import traceback
                self.reg.fail(fp, traceback.format_exc())
                log.error("[%s] eval failed: %s", fp, e)
                continue

            met, elapsed = res["metrics"], time.time() - t0
            reason = self._gate(res)
            if reason:
                self.reg.reject(fp, reason, met)
                log.info("· %-4d %s  rejected: %s", it, fp, reason)
            else:
                self.reg.complete(fp, met["score"], met, elapsed)
                self.evaluated += 1
                best = self.reg.stats()["best"] or 0.0
                flag = "  <<< NEW BEST" if met["score"] >= best - 1e-9 else ""
                log.info(
                    "· %-4d %s %-9s g%-3d score %+.4f | %4d trades | exp %+.3fR | "
                    "RR %.2f | PF %.2f | tot %+.1fR | dd %.1fR | sharpe %.2f | %.1fs%s",
                    it, fp, origin, gen, met["score"], met["trades"], met["expectancy_R"],
                    met.get("payoff_ratio", 0.0), met["profit_factor"], met["total_R"],
                    met["max_dd_R"], met["sharpe"], elapsed, flag)
                if flag:
                    R.write_strategy_report(self.cfg, self.reg, fp, res)

            if it % self.report_every == 0:
                self._checkpoint()

        self._checkpoint()
        st = self.reg.stats()
        log.info(
            "stopped after %d iterations | %d scored | %d duplicates avoided | best %.4f",
            it, self.evaluated, self.duplicates_skipped, st["best"] or 0.0)
        return st

    def _gate(self, res: dict) -> str | None:
        """Hard rejects. Recorded permanently so the genome is never retried."""
        m = res["metrics"]
        if m.get("reject_reason"):
            return m["reject_reason"]
        if res["n_folds"] < self.min_folds:
            return f"only {res['n_folds']} usable folds (need {self.min_folds})"
        if m["trades"] < self.min_trades:
            return f"only {m['trades']} trades (need {self.min_trades})"
        return None

    def _checkpoint(self):
        st = self.reg.stats()
        rate = self.evaluated / max(time.time() - self.t_start, 1e-9) * 3600
        log.info("── checkpoint: %d total, %d ok, %d rejected, %d failed | best %.4f "
                 "| %.0f eval/hr | barrier cache %d/%d",
                 st["n"], st["ok"] or 0, st["rejected"] or 0, st["failed"] or 0,
                 st["best"] or 0.0, rate, self.ev.barriers.hits,
                 self.ev.barriers.hits + self.ev.barriers.misses)
        self.reg.set_meta("last_checkpoint", time.time())
        R.write_leaderboard(self.cfg, self.reg, extra={
            "evaluated_this_run": self.evaluated,
            "duplicates_avoided": self.duplicates_skipped,
            "eval_per_hour": round(rate, 1),
            "backend": self.ev.backend,
        })
