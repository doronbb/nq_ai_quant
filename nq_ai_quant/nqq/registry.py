"""
Persistent strategy registry (SQLite, stdlib only).

Guarantees
----------
* Every genome ever *considered* is written with its fingerprint BEFORE it is
  evaluated. So a crash, a Ctrl+C, or a power cut cannot cause a repeat.
* `seen()` is an index lookup -> dedupe stays O(1) at a million strategies.
* Failures are recorded too, with the traceback, so a genome that blows up is
  never retried either.
* The DB is the single source of truth. Delete results/strategies.db to start
  the whole search over; keep it and the search resumes exactly where it was.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    fingerprint TEXT PRIMARY KEY,
    genome      TEXT NOT NULL,
    status      TEXT NOT NULL,          -- pending | ok | failed | rejected
    score       REAL,
    metrics     TEXT,
    generation  INTEGER DEFAULT 0,
    origin      TEXT,                   -- random | mutation | crossover | seed
    parents     TEXT,
    error       TEXT,
    created_at  REAL,
    finished_at REAL,
    eval_secs   REAL
);
CREATE INDEX IF NOT EXISTS ix_score  ON strategies(score DESC);
CREATE INDEX IF NOT EXISTS ix_status ON strategies(status);
CREATE INDEX IF NOT EXISTS ix_gen    ON strategies(generation);

CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL,
    config     TEXT,
    note       TEXT
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class Registry:
    def __init__(self, path: str = "results/strategies.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path, timeout=30)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        # WAL lets you open the DB in another process (e.g. the dashboard)
        # while the search is still writing.
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.commit()
        self._cache: set[str] | None = None

    # -- dedupe ------------------------------------------------------------

    def _load_cache(self) -> set[str]:
        if self._cache is None:
            self._cache = {r[0] for r in self.con.execute("SELECT fingerprint FROM strategies")}
        return self._cache

    def seen(self, fingerprint: str) -> bool:
        return fingerprint in self._load_cache()

    def reserve(self, fingerprint: str, genome: dict, origin: str,
                generation: int = 0, parents: Iterable[str] = ()) -> bool:
        """
        Claim a genome before evaluating it. Returns False if already claimed.
        Atomic, so two parallel workers cannot both take the same genome.
        """
        try:
            self.con.execute(
                "INSERT INTO strategies (fingerprint, genome, status, generation, origin,"
                " parents, created_at) VALUES (?,?,?,?,?,?,?)",
                (fingerprint, json.dumps(genome, sort_keys=True), "pending",
                 generation, origin, json.dumps(list(parents)), time.time()),
            )
            self.con.commit()
            self._load_cache().add(fingerprint)
            return True
        except sqlite3.IntegrityError:
            self._load_cache().add(fingerprint)
            return False

    # -- results -----------------------------------------------------------

    def complete(self, fingerprint: str, score: float, metrics: dict, eval_secs: float):
        self.con.execute(
            "UPDATE strategies SET status='ok', score=?, metrics=?, finished_at=?,"
            " eval_secs=? WHERE fingerprint=?",
            (float(score), json.dumps(metrics), time.time(), eval_secs, fingerprint),
        )
        self.con.commit()

    def reject(self, fingerprint: str, reason: str, metrics: dict | None = None):
        """Ran fine but failed a hard gate (too few trades, etc). Still never retried."""
        self.con.execute(
            "UPDATE strategies SET status='rejected', score=?, metrics=?, error=?,"
            " finished_at=? WHERE fingerprint=?",
            (-1e9, json.dumps(metrics or {}), reason, time.time(), fingerprint),
        )
        self.con.commit()

    def fail(self, fingerprint: str, error: str):
        self.con.execute(
            "UPDATE strategies SET status='failed', score=?, error=?, finished_at=?"
            " WHERE fingerprint=?",
            (-1e9, error[:4000], time.time(), fingerprint),
        )
        self.con.commit()

    # -- reads -------------------------------------------------------------

    def top(self, n: int = 20, min_generation: int = -1) -> list[dict]:
        rows = self.con.execute(
            "SELECT * FROM strategies WHERE status='ok' AND score IS NOT NULL"
            " AND generation > ? ORDER BY score DESC LIMIT ?",
            (min_generation, n),
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, fingerprint: str) -> dict | None:
        r = self.con.execute("SELECT * FROM strategies WHERE fingerprint=?",
                             (fingerprint,)).fetchone()
        return self._row(r) if r else None

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["genome"] = json.loads(d["genome"])
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
        d["parents"] = json.loads(d["parents"]) if d["parents"] else []
        return d

    def stats(self) -> dict[str, Any]:
        c = self.con
        row = c.execute(
            "SELECT COUNT(*) n,"
            " SUM(status='ok') ok,"
            " SUM(status='rejected') rejected,"
            " SUM(status='failed') failed,"
            " SUM(status='pending') pending,"
            " MAX(score) best,"
            " MAX(generation) gen,"
            " AVG(eval_secs) avg_secs"
            " FROM strategies"
        ).fetchone()
        return {k: row[k] for k in row.keys()}

    def cleanup_stale_pending(self, older_than_secs: float = 3600) -> int:
        """A pending row from a killed process would block that genome forever."""
        cut = time.time() - older_than_secs
        cur = self.con.execute(
            "UPDATE strategies SET status='failed', error='abandoned (process died)'"
            " WHERE status='pending' AND created_at < ?", (cut,))
        self.con.commit()
        return cur.rowcount

    def start_run(self, config: dict, note: str = "") -> int:
        cur = self.con.execute("INSERT INTO runs (started_at, config, note) VALUES (?,?,?)",
                               (time.time(), json.dumps(config, default=str), note))
        self.con.commit()
        return cur.lastrowid

    def set_meta(self, k: str, v: str):
        self.con.execute("INSERT OR REPLACE INTO meta (k,v) VALUES (?,?)", (k, str(v)))
        self.con.commit()

    def get_meta(self, k: str, default=None):
        r = self.con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def close(self):
        self.con.close()
