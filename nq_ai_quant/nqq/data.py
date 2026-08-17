"""
Data layer. Pluggable adapters -> a single canonical bar DataFrame.

Canonical format:
    index : tz-aware DatetimeIndex in America/New_York, ascending, unique
    cols  : open, high, low, close, volume   (float64)

Adapters (set in config.yaml -> data.source):
    csv        : read a folder / file of OHLCV CSVs. Most reliable. No API key.
    databento  : GLBX.MDP3 NQ continuous front month. Best production quality.
    yfinance   : free bootstrap feed. ~60 days of intraday only. Prototyping only.
"""
from __future__ import annotations

import glob
import gzip
import os
import pickle
import hashlib
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("nqq.data")

NY = "America/New_York"
OHLCV = ["open", "high", "low", "close", "volume"]

# Pandas resample aliases keyed by our timeframe strings.
_TF_RULE = {
    "1min": "1min", "2min": "2min", "3min": "3min", "5min": "5min",
    "10min": "10min", "15min": "15min", "30min": "30min",
    "60min": "60min", "1h": "60min", "4h": "240min", "1d": "1D",
}

# Minutes per bar, used for horizon/embargo math.
_TF_MINUTES = {
    "1min": 1, "2min": 2, "3min": 3, "5min": 5, "10min": 10, "15min": 15,
    "30min": 30, "60min": 60, "1h": 60, "4h": 240, "1d": 1440,
}


def tf_minutes(tf: str) -> int:
    if tf not in _TF_MINUTES:
        raise ValueError(f"unknown timeframe {tf!r}")
    return _TF_MINUTES[tf]


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "o": "open", "h": "high", "l": "low", "c": "close",
    "v": "volume", "vol": "volume", "vwap": "vwap",
    "tickvolume": "tick_volume", "tick_vol": "tick_volume",
    "ticks": "tick_volume", "trades": "tick_volume", "count": "tick_volume",
    "date": "timestamp", "datetime": "timestamp", "time": "timestamp",
    "ts": "timestamp", "ts_event": "timestamp", "date_time": "timestamp",
    "px_open": "open", "px_high": "high", "px_low": "low", "px_close": "close",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    return df


def _finalize(df: pd.DataFrame, source_tz: str) -> pd.DataFrame:
    """Coerce any adapter output into the canonical format."""
    df = _normalize_columns(df)

    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
        df = df.drop(columns=["timestamp"])
    else:
        idx = pd.to_datetime(df.index, errors="coerce")

    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
    df.index = idx.tz_convert(NY)

    # Many retail exports (MT4/MT5, some brokers) ship real volume as a column of
    # zeros and put the only usable activity measure in a tick-count column.
    # A volume series that is identically zero is worse than useless: it silently
    # zeroes every volume-derived feature. Fall back to tick counts when that happens.
    if "tick_volume" in df.columns:
        vol = pd.to_numeric(df.get("volume"), errors="coerce") if "volume" in df.columns else None
        if vol is None or float(np.nansum(np.abs(vol.to_numpy(dtype="float64")))) == 0.0:
            log.warning("volume column is empty/zero - using tick volume instead")
            df["volume"] = df["tick_volume"]
        df = df.drop(columns=["tick_volume"])

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"data source is missing required columns: {missing}")

    df = df[OHLCV].astype("float64")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Drop bars with impossible geometry (bad ticks / vendor glitches).
    ok = (
        df[OHLCV[:4]].notna().all(axis=1)
        & (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9)
        & (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9)
        & (df["close"] > 0)
    )
    dropped = int((~ok).sum())
    if dropped:
        log.warning("dropped %d malformed bars", dropped)
    df = df[ok]
    df["volume"] = df["volume"].fillna(0.0)
    return df


def _sniff_sep(path: str) -> str:
    """Pick the delimiter from the header line. Tab- and semicolon-separated
    exports are common and pandas' comma default reads them as one fat column."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    counts = {sep: head.count(sep) for sep in ("\t", ";", ",", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 3 else ","


def csv_files(path: str) -> list[str]:
    """Every data file under `path`, whether it is a folder, a file or a glob."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        pats = ("*.csv", "*.csv.gz", "*.txt", "*.tsv")
        return sorted(
            f for pat in pats
            for f in glob.glob(os.path.join(path, "**", pat), recursive=True)
        )
    return sorted(glob.glob(path))


def _median_spacing_minutes(idx: pd.DatetimeIndex) -> float:
    """Native bar interval of a file, robust to weekend and holiday gaps."""
    if len(idx) < 3:
        return 0.0
    # Via total_seconds rather than the integer view: pandas indexes are not
    # always nanosecond-resolution (microsecond is the default in newer
    # versions), and assuming ns silently scales the answer by 1000.
    d = idx.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else 0.0


def _read_one(f: str, source_tz: str) -> pd.DataFrame:
    raw = pd.read_csv(f, sep=_sniff_sep(f))
    # Headerless files: assume timestamp,o,h,l,c,v ordering.
    if raw.shape[1] >= 6 and not any(
        str(c).strip().lower() in _COLUMN_ALIASES or str(c).strip().lower() in OHLCV
        for c in raw.columns
    ):
        raw = pd.read_csv(f, header=None, sep=_sniff_sep(f))
        raw = raw.iloc[:, :6]
        raw.columns = ["timestamp"] + OHLCV
    return _finalize(raw, source_tz)


def _load_csv(cfg: dict, timeframe: str = "15min") -> pd.DataFrame:
    path = os.path.expanduser(cfg["path"])
    files = csv_files(path)
    if not files:
        raise FileNotFoundError(
            f"No data files found under {path!r}. Drop your NQ OHLCV CSVs in there "
            f"(.csv, .csv.gz, .tsv or .txt) and run again."
        )

    target = tf_minutes(timeframe)
    log.info("reading %d data file(s) from %s", len(files), path)
    frames = []
    for f in files:
        try:
            df = _read_one(f, cfg.get("source_tz", NY))
        except Exception as e:
            log.warning("skipping %s — could not parse it (%s)", os.path.basename(f), e)
            continue
        if df.empty:
            log.warning("skipping %s — no usable bars in it", os.path.basename(f))
            continue

        # A file coarser than the target timeframe cannot be resampled up to it,
        # and silently mixing daily bars into a 15-minute frame would produce a
        # series that looks fine and is nonsense. Refuse it out loud instead.
        spacing = _median_spacing_minutes(df.index)
        if spacing > target * 1.5:
            log.warning(
                "skipping %s — its bars are ~%.0f min apart, coarser than the "
                "%s target. Lower data.timeframe to use this file.",
                os.path.basename(f), spacing, timeframe)
            continue

        log.info("  %-28s %7d bars  ~%.0fmin  %s -> %s", os.path.basename(f), len(df),
                 spacing, df.index[0].date(), df.index[-1].date())
        frames.append(df)

    if not frames:
        raise ValueError(
            f"None of the {len(files)} file(s) under {path!r} are usable at "
            f"timeframe={timeframe}. See the warnings above.")

    out = pd.concat(frames).sort_index()
    # Overlapping files (a fresh export on top of an old one) resolve to the
    # most recently sorted value rather than duplicating the bar.
    dupes = int(out.index.duplicated().sum())
    if dupes:
        log.info("%d overlapping bars across files, keeping one of each", dupes)
    return out[~out.index.duplicated(keep="last")]


def _load_databento(cfg: dict, timeframe: str = "15min") -> pd.DataFrame:
    try:
        import databento as db
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install databento  (or switch data.source to 'csv')") from e

    key = cfg.get("api_key") or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise ValueError("set data.api_key in config.yaml or the DATABENTO_API_KEY env var")

    client = db.Historical(key)
    schema = {"1min": "ohlcv-1m", "1h": "ohlcv-1h", "60min": "ohlcv-1h", "1d": "ohlcv-1d"}.get(
        cfg.get("native_timeframe", "1min"), "ohlcv-1m"
    )
    log.info("databento pull %s %s -> %s", cfg["start"], cfg["end"], schema)
    store = client.timeseries.get_range(
        dataset=cfg.get("dataset", "GLBX.MDP3"),
        symbols=cfg.get("symbol", "NQ.c.0"),
        stype_in=cfg.get("stype_in", "continuous"),
        schema=schema,
        start=cfg["start"],
        end=cfg["end"],
    )
    df = store.to_df()
    if df.index.name != "ts_event" and "ts_event" in df.columns:
        df = df.set_index("ts_event")
    return _finalize(df, "UTC")


def _load_yfinance(cfg: dict, timeframe: str = "15min") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install yfinance  (or switch data.source to 'csv')") from e

    interval = cfg.get("native_timeframe", "15min").replace("min", "m").replace("1h", "60m")
    period = cfg.get("period", "60d")
    log.warning(
        "yfinance gives ~%s of intraday NQ=F only, with gaps. Use it to smoke-test, "
        "not to draw conclusions.", period
    )
    df = yf.download(
        cfg.get("symbol", "NQ=F"), period=period, interval=interval,
        progress=False, auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _finalize(df.reset_index(), "UTC")


_ADAPTERS = {"csv": _load_csv, "databento": _load_databento, "yfinance": _load_yfinance}


# --------------------------------------------------------------------------
# resampling / sessions / public entrypoint
# --------------------------------------------------------------------------

def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Downsample to `timeframe`. Never upsamples (that would invent data)."""
    rule = _TF_RULE.get(timeframe)
    if rule is None:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    out = (
        df.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out


def tag_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add RTH / overnight session flags. NQ RTH = 09:30-16:00 New York."""
    idx = df.index
    mod = idx.hour * 60 + idx.minute                       # minute of day, NY
    df = df.copy()
    df["_minute_of_day"] = mod
    df["_is_rth"] = ((mod >= 570) & (mod < 960)).astype(np.int8)          # 09:30-16:00
    df["_is_open_drive"] = ((mod >= 570) & (mod < 630)).astype(np.int8)   # first 60m
    df["_is_close"] = ((mod >= 900) & (mod < 960)).astype(np.int8)        # last 60m
    df["_is_globex"] = (1 - df["_is_rth"]).astype(np.int8)
    df["_dow"] = idx.dayofweek.astype(np.int8)
    df["_session_date"] = pd.Series(
        np.where(mod >= 1080, idx.date + pd.Timedelta(days=1), idx.date), index=idx
    )
    return df


def _cache_key(cfg: dict, timeframe: str) -> str:
    blob = repr(sorted((k, str(v)) for k, v in cfg.items())) + "|" + timeframe
    if cfg.get("source") == "csv":
        # The config alone does not identify the data when `path` is a folder you
        # drop files into: adding, replacing or extending a CSV leaves every
        # config value untouched. Without the file signature in the key you would
        # drop in a year of new bars, see "bars from cache", and search the old
        # data forever.
        sig = [(os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f)))
               for f in csv_files(cfg.get("path", ""))]
        blob += "|" + repr(sig)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def load_bars(cfg: dict, timeframe: str, cache_dir: str = "results/cache",
              use_cache: bool = True) -> pd.DataFrame:
    """Load, resample, session-tag and cache bars. This is the only entrypoint you need."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"bars_{_cache_key(cfg, timeframe)}.pkl")

    if use_cache and os.path.exists(cache_path):
        max_age = float(cfg.get("cache_max_age_hours", 12))
        age_h = (pd.Timestamp.utcnow().timestamp() - os.path.getmtime(cache_path)) / 3600
        if age_h < max_age or cfg["source"] == "csv":
            with open(cache_path, "rb") as fh:
                log.info("bars from cache (%s)", os.path.basename(cache_path))
                return pickle.load(fh)

    source = cfg["source"]
    if source not in _ADAPTERS:
        raise ValueError(f"unknown data.source {source!r}; pick one of {list(_ADAPTERS)}")

    raw = _ADAPTERS[source](cfg, timeframe)
    if raw.empty:
        raise ValueError("data source returned zero bars")

    df = resample(raw, timeframe)

    start, end = cfg.get("start"), cfg.get("end")
    if start:
        df = df[df.index >= pd.Timestamp(start).tz_localize(NY) if pd.Timestamp(start).tz is None
                else df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end).tz_localize(NY) if pd.Timestamp(end).tz is None
                else df.index <= pd.Timestamp(end)]

    df = tag_sessions(df)
    log.info("loaded %d %s bars  %s -> %s", len(df), timeframe,
             df.index[0].date(), df.index[-1].date())

    with open(cache_path, "wb") as fh:
        pickle.dump(df, fh, protocol=4)
    return df
