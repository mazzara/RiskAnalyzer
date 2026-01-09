# src/cache/cash_yf.py

import os, json, time 
from contextlib import contextmanager 
from datetime import datetime, timedelta 
import pandas as pd 
from pathlib import Path 

CACHE_ROOT = Path('data_cache/yf') 
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# Monthly files for granularity - exemple: 1 minute uses monthly files
_GRANULAR = {"1m": "month", "2m": "month", "5m": "month", "15m": "month",
             "30m": "month", "60m": "month", "1h": "month", "90m": "month",
             "4h": "month"}
def _grain(interval): return _GRANULAR.get(interval.lower(), "year")

def _periods_between(start: pd.Timestamp, end: pd.Timestamp, interval: str):
    if _grain(interval) == "month":
        current = pd.Timestamp(start.year, start.month, 1)
        while current <= end:
            yield ("month", current.year, current.month)
            current += pd.offsets.MonthBegin(1)
    else:
        for y in range(start.year, end.year +1):
            yield ("year", y, None)


def _cache_dir(symbol: str, interval: str) -> Path:
    return CACHE_ROOT / symbol.replace("/", "_") / interval


# def _cache_path(symbol: str, 
#                 interval: str,
#                 year: int, month: int = None) -> Path:
#     d = _cache_dir(symbol, interval)
#     d.mkdir(parents=True, exist_ok=True)
#     if month:
#         return d / f"{year}-{month:02d}.parquet"
#     return d / f"{year}.parquet"

def _cache_paths(symbol: str, interval: str):
    base = CACHE_ROOT / symbol / interval
    _ensure_dir(base)
    return base / "data.parquet", base / "meta.json"


def _read_cache(pq_path: Path) -> pd.DataFrame:
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        return _normalize_df_dates(df)
    return pd.DataFrame()


def _write_cache(pq_path: Path, df: pd.DataFrame):
    df = _normalize_df_dates(df)
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq_path, index=False)


def _manifest_path(symbol: str, interval: str) -> Path:
    return _cache_dir(symbol, interval) / "manifest.json"

# @contextmanager
# def _lock(symbol: str, interval: str):
#     lock_path = _cache_dir(symbol, interval) / ".lock"
#     while True:
#         try:
#             fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
#             os.close(fd); break
#         except FileExistsError:
#             time.sleep(0.1)
#
#     try:
#         yield
#     finally:
#         try:
#             os.remove(lock_path)
#         except FileNotFoundError: pass 

# @contextmanager
# def _lock(symbol, interval):
#     # make sure the cache dir exists before creating the lock file
#     _cache_dir(symbol, interval).mkdir(parents=True, exist_ok=True)
#     lock_path = _cache_dir(symbol, interval) / ".lock"
#     while True:
#         try:
#             fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
#             os.close(fd)
#             break
#         except FileExistsError:
#             time.sleep(0.1)
#     try:
#         yield
#     finally:
#         try:
#             os.remove(lock_path)
#         except FileNotFoundError:
#             pass


@contextmanager
def _lock(symbol: str, interval: str):
    # make sure folder exists before creating the lock file
    sym_dir = CACHE_ROOT / symbol / interval
    _ensure_dir(sym_dir)  # <— important
    lock_path = sym_dir / ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, b"1")
        yield
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _to_naive_utc(s):
    # robust normalize -> tz-aware UTC -> tz-naive
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None) if hasattr(s, "dt") else s.tz_convert(None)


def _normalize_df_dates(df: pd.DataFrame) -> pd.DataFrame:
    # unify expected column names and datatypes
    if "date" not in df.columns:
        # yfinance returns Date or Datetime → rename to 'date'
        for cand in ("Date", "Datetime", "datetime"):
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    df["date"] = _to_naive_utc(df["date"])
    # guarantee adj_close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    return df



def _read_manifest(symbol, interval):
    p = _manifest_path(symbol, interval)
    if not p.exists():
        return {"last_refresh": None}
    try:
        with p.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"last_refresh": None}

def _write_manifest(symbol, interval, m):
    p = _manifest_path(symbol, interval)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(m, f)
    os.replace(tmp, p)

def _load_piece(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    return pd.read_parquet(path)

def _save_piece_atomic(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    # normalize names
    if "Date" in df.columns or "Datetime" in df.columns:
        df = df.rename(columns={"Date":"date","Datetime":"date"})
    df["date"] = pd.to_datetime(df["date"])
    # normalize date utc-aware
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    req = ["date","open","high","low","close","adj_close","volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:  # fill missing numeric with NaN-safe defaults
        for c in missing:
            df[c] = pd.NA
    df = df[req].dropna(subset=["date"]).sort_values("date")
    df = df[~df["date"].duplicated(keep="last")]
    return df


#
# def fetch_with_cache(symbol: str, start: str, end: str, interval: str, fetch_func):
#     """
#     fetch_func(start_iso, end_iso, interval) -> DataFrame with at least date/open/high/low/close/volume
#     """
#     start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
#     pieces = []
#     _cache_dir(symbol, interval).mkdir(parents=True, exist_ok=True)
#     with _lock(symbol, interval):
#         # load existing cached pieces
#         for grain, y, m in _periods_between(start_ts, end_ts, interval):
#             p = _cache_path(symbol, interval, y, m)
#             df = _load_piece(p)
#             if not df.empty: pieces.append(df)
#
#         have_coverage = False
#         if pieces:
#             merged = pd.concat(pieces, ignore_index=True)
#             merged = _ensure_schema(merged)
#             have_coverage = (merged["date"].min() <= start_ts) and (merged["date"].max() >= end_ts)
#         else:
#             merged = pd.DataFrame()
#
#         if not have_coverage:
#             # download missing chunks per period
#             for grain, y, m in _periods_between(start_ts, end_ts, interval):
#                 p = _cache_path(symbol, interval, y, m)
#                 # compute sub-range to fetch
#                 if grain == "month":
#                     a = pd.Timestamp(y, m, 1)
#                     b = (a + pd.offsets.MonthBegin(2)) - pd.offsets.Day(1)  # end-of-month
#                 else:
#                     a = pd.Timestamp(y, 1, 1)
#                     b = pd.Timestamp(y, 12, 31)
#                 a = max(a, start_ts); b = min(b, end_ts)
#                 if a > b: continue
#                 # skip if we already have this file covering a..b
#                 cached = _load_piece(p)
#                 if not cached.empty:
#                     cached = _ensure_schema(cached)
#                     if cached["date"].min() <= a and cached["date"].max() >= b:
#                         continue
#                 # fetch and save
#                 part = fetch_func(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), interval)
#                 part = _ensure_schema(part)
#                 if part is None or part.empty: 
#                     continue
#                 _save_piece_atomic(part, p)
#
#             # reload all pieces covering requested window
#             pieces = []
#             for grain, y, m in _periods_between(start_ts, end_ts, interval):
#                 p = _cache_path(symbol, interval, y, m)
#                 df = _load_piece(p)
#                 if not df.empty: pieces.append(df)
#             merged = _ensure_schema(pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame())
#
#         # small recent refresh: refetch last N days into the last piece
#         N = 5
#         tail_start = (end_ts - timedelta(days=N)).strftime("%Y-%m-%d")
#         tail = _ensure_schema(fetch_func(tail_start, end, interval))
#         if not tail.empty:
#             merged = _ensure_schema(pd.concat([merged, tail], ignore_index=True))
#
#         # cut to requested window
#         final = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)].copy()
#         _write_manifest(symbol, interval, {"last_refresh": datetime.utcnow().isoformat()+"Z"})
#         return final.reset_index(drop=True)



def fetch_with_cache(symbol: str, interval: str, start: str, end: str, fetch_fn):
    """
    fetch_fn(start_iso, end_iso, interval) -> DataFrame with 'date' column (tz-naive).
    """
    pq_path, meta_path = _cache_paths(symbol, interval)

    with _lock(symbol, interval):
        existing = _read_cache(pq_path)

        # Normalize request bounds to tz-naive UTC
        start_ts = _to_naive_utc(pd.Timestamp(start))
        end_ts   = _to_naive_utc(pd.Timestamp(end))

        # Detect what’s missing
        have_min = existing["date"].min() if not existing.empty else None
        have_max = existing["date"].max() if not existing.empty else None

        need_head = have_min is None or start_ts < have_min
        need_tail = have_max is None or end_ts > have_max

        # download head
        head = pd.DataFrame()
        if need_head:
            head_end = have_min if have_min is not None else end_ts
            head = fetch_fn(start_ts.strftime("%Y-%m-%d"), head_end.strftime("%Y-%m-%d"), interval)

        # download tail
        tail = pd.DataFrame()
        if need_tail:
            tail_start = have_max if have_max is not None else start_ts
            tail = fetch_fn(tail_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), interval)

        # merge
        frames = [x for x in (existing, head, tail) if x is not None and not x.empty]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if not merged.empty:
            merged = _normalize_df_dates(merged)
            merged = merged.sort_values("date").drop_duplicates("date")
            _write_cache(pq_path, merged)

        # final slice to exactly the requested window
        if merged.empty:
            return merged

        final = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)].copy()
        return final
