# ./src/datasources/yfinance_fetcher.py 

import pandas as pd
import yfinance as yf
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from datetime import datetime, timedelta
from src.logger_config import logger
import time, random
from yfinance.exceptions import YFRateLimitError
from src.cache.cache_yf import fetch_with_cache
_last_call = 0.0

def _throttle(min_delay=0.8, max_delay=1.6):
    """Sleep a bit between Yahoo calls to reduce 429s. Jitter helps."""
    global _last_call
    now = time.monotonic()
    # enforce min spacing
    gap = now - _last_call
    if gap < min_delay:
        time.sleep(min_delay - gap)
    # light jitter occasionally
    if random.random() < 0.35:
        time.sleep(random.uniform(0.0, max_delay - min_delay))
    _last_call = time.monotonic()


# Yahoo Finance limitations for intraday intervals
INTRADAY_LIMITS = {
    '1m': 7,
    '2m': 60,
    '5m': 60,
    '15m': 60,
    '30m': 60,
    '60m': 730,  # 2 years
    '1h': 730,  # 2 years
    '90m': 60,
}


INTERVAL_ALIASES = {
    '1h': '60m',
    '60min': '60m',
}

def _norm_interval(interval: str) -> str:
    if not interval:
        return "1d"
    i = interval.strip().lower()
    return INTERVAL_ALIASES.get(i, i)

# Set up a requests session with a custom header to avoid blocking
# session = requests.Session()
# session.headers.update({
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
# })
#

_SESSION = None
def get_session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_4) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
            )
        })
        retry = Retry(
            total=3, backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"], raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        _SESSION = s
    return _SESSION



def _download_with_retry(symbol, start_date, end_date, interval, auto_adjust, max_retries=4):
    # s = get_session()
    for attempt in range(1, max_retries+1):
        try:
            _throttle()  # be nice to Yahoo
            return yf.download(
                symbol, 
                start=start_date, 
                end=end_date,
                interval=interval, 
                # session=s,
                progress=False, 
                auto_adjust=auto_adjust,
            )
        except YFRateLimitError:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.uniform(0.3, 1.7)
            logger.warning(f"[yfinance] 429 rate-limited (attempt {attempt}/{max_retries}). Sleeping {sleep_s:.1f}s…")
            time.sleep(sleep_s)



def _normalize_yf(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # unify time col
    for cand in ("Date", "Datetime", "date", "datetime"):
        if cand in df.columns:
            if cand != "date":
                df = df.rename(columns={cand: "date"})
            break
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)

    # --- FIX: flatten MultiIndex columns (yfinance 1.0) ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
                c[0] if isinstance(c, tuple) else c 
                for c in df.columns 
        ]

    # lower names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # adj_close guarantee
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # handle multiindex (rare)
    # if isinstance(df.columns, pd.MultiIndex):
    #     df.columns = ["_".join([str(x) for x in t if x]) for t in df.columns]


    # strip symbol suffixes if any
    sym = symbol.lower()
    remap = {
        f"open_{sym}":"open", f"high_{sym}":"high",
        f"low_{sym}":"low",   f"close_{sym}":"close",
        f"adj_close_{sym}":"adj_close",
        f"volume_{sym}":"volume",
    }
    df = df.rename(columns=remap)
    return df[["date","open","high","low","close","adj_close","volume"]].dropna(subset=["open","high","low","close"])





def _fetch_chart_direct(symbol, start_ts, end_ts, interval):
    s = get_session()
    url = "https://query2.finance.yahoo.com/v8/finance/chart/" + symbol
    params = {
        "period1": int(pd.Timestamp(start_ts).timestamp()),
        "period2": int(pd.Timestamp(end_ts).timestamp()),
        "interval": interval,                 # e.g., '1d', '15m', '30m', '60m'
        "includePrePost": False,
        "events": "div,splits,capitalGains",
    }
    _throttle()  # be nice to Yahoo
    r = s.get(url, params=params, timeout=(3.05, 20))  # reuse your session w/ UA
    if r.status_code == 429:
        raise YFRateLimitError("429 on direct chart endpoint")
    r.raise_for_status()
    j = r.json()

    res = j.get("chart", {}).get("result")
    if not res:
        return pd.DataFrame()

    res = res[0]
    ts = res.get("timestamp", [])
    if not ts:
        return pd.DataFrame()

    q = res.get("indicators", {}).get("quote", [{}])[0]
    df = pd.DataFrame({
        # "date": pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None),
        # UTC aware
        "date": pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_localize(None),
        "open": q.get("open", []),
        "high": q.get("high", []),
        "low":  q.get("low", []),
        "close":q.get("close", []),
        "volume": q.get("volume", []),
    })
    df = df.dropna(subset=["open","high","low","close"])  # robust
    return df


# def _raw_fetch(start_iso, end_iso, interval):
#     data = _download_with_retry(
#         symbol,
#         start_iso,
#         end_iso,
#         interval,
#         auto_adjust=False,
#         max_retries=4
#     )
#     return data

def _fetch_range(symbol: str, start_iso: str, end_iso: str, 
                 interval: str, auto_adjust: bool) -> pd.DataFrame:
    """Fetch one window, normalize columns, return a uniform DF (date, ohlcv, adj_close)."""
    interval = _norm_interval(interval)
    start_iso, end_iso = validate_date_range(interval, start_iso, end_iso)
    try:
        data = _download_with_retry(
            symbol,
            start_iso,
            end_iso,
            interval,
            auto_adjust,
            max_retries=4
        )
    except YFRateLimitError as e:
        logger.warning(f"[yfinance] 429 rate-limited; trying direct chart fallback. {e}")
        data = _fetch_chart_direct(symbol, start_iso, end_iso, interval)

    if data is None or data.empty:
        return pd.DataFrame()

    # --- normalize time axis
    data = data.copy()
    data.reset_index(inplace=True)
    time_col = next((c for c in ('Date', 'Datetime', 'date', 'datetime') if c
        in data.columns), None)
    if time_col is None:
        data["date"] = data.index
    else:
        if time_col != "date":
            data.rename(columns={time_col: "date"}, inplace=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    # force single standard UTC-aware 
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_localize(None)

    # --- flatten & tidy 
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(lv) for lv in col if lv]) for col in data.columns]
    data.columns = [c.lower().replace(" ", "_") for c in data.columns]

    # chart fallback lacks adj_close -> synthesize
    if "adj_close" not in data.columns and "close" in data.columns:
        data["adj_close"] = data["close"]

    # standarize OHLCV names with suffix maps
    data = normalize_columns(data, symbol)

    # final sanity
    data = data.dropna(subset=["date", "open", "high", "low", "close"])
    data = data.sort_values("date").reset_index(drop=True)

    return data


def validate_date_range(interval, start_date, end_date):
    """
    Validate date ranges based on Yahoo Finance interval limitations.
    If invalid, return an adjusted start_date or raise a warning.
    """
    if interval in INTRADAY_LIMITS:
        limit_days = INTRADAY_LIMITS[interval]

        # Convert to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        max_start_dt = end_dt - timedelta(days=limit_days)

        if start_dt < max_start_dt:
            print(f"[Warning] Interval '{interval}' only allows {limit_days} days of history.")
            print(f"[Action] Adjusting start date from {start_date} to {max_start_dt.date()}")
            start_dt = max_start_dt
        
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    
    return start_date, end_date


def normalize_columns(data, symbol):
    """
    Renames columns with symbol-specific suffixes to generic OHLCV columns.

    E.g., from 'close_ng=f' to 'close'.
    """
    symbol_prefix = symbol.lower()

    rename_map = {
        f'open_{symbol_prefix}': 'open',
        f'high_{symbol_prefix}': 'high',
        f'low_{symbol_prefix}': 'low',
        f'close_{symbol_prefix}': 'close',
        f'adj_close_{symbol_prefix}': 'adj_close',
        f'volume_{symbol_prefix}': 'volume'
    }

    # Perform rename
    data.rename(columns=rename_map, inplace=True)

    # Optional sanity check
    # required_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    required_cols = ['open', 'high', 'low', 'close', 'volume'] # adj_close may be synthesized earlier
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"[Error] Missing expected columns after renaming: {missing}")

    return data



def get_market_data(symbol, start_date, end_date, interval="1d", auto_adjust=False):
    logger.info(f"[ 3705:05:05 :: Fetch] Yahoo URL: ... (symbol={symbol} interval={interval})")
    # fetch_fn must return a normalized DF with 'date' tz-naive and OHLCV
    def fetch_fn(a, b, i):
        raw = _download_with_retry(symbol, a, b, i, auto_adjust=auto_adjust, max_retries=4)
        return _normalize_yf(raw, symbol)

    data = fetch_with_cache(
        symbol=symbol,
        interval=interval,
        start=start_date,
        end=end_date,
        fetch_fn=fetch_fn
    )
    if data is None or data.empty:
        raise ValueError(f"[yfinance] Empty data for {symbol} {interval} {start_date}->{end_date}")

    # tiny debug
    logger.debug(f"[yfinance] fetched rows={len(data)} range={data['date'].min()}→{data['date'].max()}")
    return data




# def get_market_data(symbol, start_date, end_date, interval='1d', auto_adjust=False):
#     """
#     Fetch historical market data for a specified symbol from Yahoo Finance.
#
#     Parameters:
#         symbol (str): Yahoo Finance ticker symbol (e.g., 'CL=F' for Crude Oil Futures).
#         start_date (str): Start date in 'YYYY-MM-DD' format.
#         end_date (str): End date in 'YYYY-MM-DD' format.
#         interval (str): Data frequency ('1d', '1wk', '1mo', etc.). Default is '1d'.
#
#     Returns:
#         pd.DataFrame: DataFrame containing historical market data.
#     """
#
#     # Validate dates for intraday intervals
#     start_date, end_date = validate_date_range(interval, start_date, end_date)
#
#     # Normalize & validate once for the top-level window (cache will also validade pre-slices)
#     interval = _norm_interval(interval)
#     start_date, end_date = validate_date_range(interval, start_date, end_date)
#
#     try:
#         # Build a reproducible query string for debugging
#         base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
#         query_params = {
#             "symbol": symbol,
#             "interval": interval,
#             "period1": int(pd.Timestamp(start_date).timestamp()),
#             "period2": int(pd.Timestamp(end_date).timestamp()),
#             "events": "div,splits",
#             "includeAdjustedClose": "true" if auto_adjust else "false",
#         }
#         debug_url = f"{base_url}{symbol}?{urlencode(query_params)}"
#         logger.info(f"[ 3705:05:05 :: Fetch] Yahoo URL: {debug_url}") 
#         print(f"[Info] Fetching {interval} data for {symbol} from {start_date} to {end_date}")
#         # Fetch data from Yahoo Finance
#         # data = yf.download(
#         #     symbol,
#         #     start=start_date,
#         #     end=end_date,
#         #     interval=interval,
#         #     session=session,
#         #     progress=False,
#         #     auto_adjust=auto_adjust
#         # )
#
#         # Alternative data fetcher with helper function  
#         data = _download_with_retry(
#             symbol,
#             start_date,
#             end_date,
#             interval,
#             auto_adjust,
#             max_retries=4
#         )
#
#     except YFRateLimitError as e:
#         logger.warning(f"[3705:05:10 yfinance] 429 rate-limited; trying direct chart fallback. {e}")
#         data = _fetch_chart_direct(symbol, start_date, end_date, interval)
#
#     if data is None or data.empty:
#         msg = (f"[yfinance] Empty data for {symbol} ({interval}) {start_date}→{end_date} "
#                f"after retry/fallback. Likely 429 or out-of-range.")
#         logger.error(msg)
#         raise ValueError(msg)
#
#     # Log  
#     base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
#     query_params = {
#             "symbol": symbol, 
#             "interval": interval,
#             "period1": int(pd.Timestamp(start_date).timestamp()),
#             "period2": int(pd.Timestamp(end_date).timestamp()),
#             "events": "div,splits",
#             "includeAdjustedClose": "true" if auto_adjust else "false",
#     }
#     debug_url = f"{base_url}{symbol}?{urlencode(query_params)}"
#     logger.info(f"[ 3705:05:05 :: Fetch] Yahoo URL: {debug_url}")
#
#     # Use the cache to assemble full window from smaller slices
#     data = fetch_with_cache(
#         symbol=symbol,
#         start=start_date,
#         end=end_date,
#         interval=interval,
#         fetch_func=lambda a, b, i: _fetch_range(symbol, a, b, i, auto_adjust)
#     )
#
#     # except YFRateLimitError as e:
#     #     logger.warning(f"[3705:05:10 yfinance] Rate-limited; trying direct chart fallback. {e}")
#     #     data = _fetch_chart_direct(symbol, start_date, end_date, interval)
#     # if data.empty:
#     #     raise
#
#
#     # except Exception as e:
#     #     logger.error(f"[3705:05:10 Error] Error fetching data for {symbol}: {e}")
#     #     print(f"[3705:05:10 Error] Fetching data for {symbol}: {e}")
#     #     return pd.DataFrame()
#
#     # if data is None or data.empty:
#     #     logger.warning(f"[3705:05:15 Warning] No data retrieved for {symbol} from Yahoo Finance.")
#     #     print(f"[3705:05:20 Warning] No data retrieved for {symbol}. Check symbol and date range.")
#     #     return pd.DataFrame()
#
#     if data is None or data.empty:
#         msg = (f"[3705:05:15 yfinance] No data for {symbol} ({interval}) "
#                f"{start_date} → {end_date}. Possibly rate-limited or out-of-range.")
#         logger.error(msg)
#         return pd.DataFrame()
#
#     # (No second empty-check; the one above is enough)
#
#     # Reset index to turn datetime index into a column
#     data.reset_index(inplace=True)
#
#     # Normalize column to 'date' ---------------------
#     # time_col = None
#     # for candidate in ['Date', 'Datetime']:
#     #     if candidate in data.columns:
#     #         time_col = candidate
#     #         break
#     #
#     # if time_col is None:
#     #     print(f"[3705:05:30 Warning] No Date/Datetime column found. Using index as 'date'.")
#     #     data['date'] = data.index
#     # else:
#     #     data.rename(columns={time_col: 'date'}, inplace=True)
#
#
#
#     # Normalize time column to 'date' -----------------
#     time_col = None
#     for candidate in ['Date', 'Datetime', 'date', 'datetime']:
#         if candidate in data.columns:
#             time_col = candidate
#             break
#     if time_col is None:
#         logger.warning("[3705:05:30 Warning] No Date/Datetime column found. Using index as 'date'.")
#         data['date'] = data.index
#     else:
#         if time_col != 'date':
#             data.rename(columns={time_col: 'date'}, inplace=True)
#
#     # Ensure 'date' is in datetime format
#     # data['date'] = pd.to_datetime(data['date'])
#     data['date'] = pd.to_datetime(data['date'], errors='coerce')
#     logger.debug(f"[3705:10:05 Debug] Data columns before normalization: {data.columns.tolist()}")
#
#     # Handle MultiIndex column names (flatten if necessary)
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = ['_'.join([str(level) for level in col if level]) for col in data.columns]
#
#     # Ensure all column names are lowercase and spaces replaced
#     data.columns = [col.lower().replace(' ', '_') for col in data.columns]
#
#     # Ensure 'adj_close' exists (chart fallback doesn't provide it)
#     if 'adj_close' not in data.columns and 'close' in data.columns:
#         data['adj_close'] = data['close']
#
#     data = normalize_columns(data, symbol)
#     logger.debug(f"[3705:10:07 Debug] Data columns after normalization: {data.columns.tolist()}")
#
#     # Display first few rows for verification
#     print(f"[Info] Data fetched successfully for {symbol}:\n")
#     logger.debug(f"[3705:10:10 Debug] Sample data:\n{data.head()}")
#
#     # [Info]: Show dataset description
#     print("\n[Info] Dataset Metadata:")
#     print(f"- Timeframe: {interval}")
#     print(f"- Date Range: {start_date} to {end_date}")
#     print(f"- Rows: {len(data)}")
#     print(f"- Columns ({len(data.columns)}): {list(data.columns)}")
#     print(f"- First Date: {data['date'].iloc[0] if not data.empty else 'N/A'}")
#     print(f"- Last Date: {data['date'].iloc[-1] if not data.empty else 'N/A'}")
#
#     # Preview first and last 3 rows (optional for deeper debugging)
#     print("\n[Preview] First 3 rows:\n", data.head())
#     print("\n[Preview] Last 3 rows:\n", data.tail())
#
#     return data


# End of data_fetcher.py

