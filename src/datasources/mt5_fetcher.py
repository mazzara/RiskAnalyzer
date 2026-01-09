# ./src/datasources/mt5_fetcher.py

# ./src/datasources/mt5_fetcher.py
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("[Warning] MetaTrader5 module not found. Running in non-MT5 mode.")

import pandas as pd
from datetime import datetime
from src.datasources import mt5_connect

# Build the map only if mt5 is available
if mt5:
    INTERVAL_MAP = {
        '1m':  mt5.TIMEFRAME_M1,
        '5m':  mt5.TIMEFRAME_M5,
        '15m': mt5.TIMEFRAME_M15,
        '30m': mt5.TIMEFRAME_M30,
        '1h':  mt5.TIMEFRAME_H1,
        '4h':  mt5.TIMEFRAME_H4,
        '1d':  mt5.TIMEFRAME_D1,
        '1w':  mt5.TIMEFRAME_W1,
        '1mo': mt5.TIMEFRAME_MN1,
    }
else:
    # Fallback stub so rest of code still works
    INTERVAL_MAP = {}

# Global connection flag
MT5_CONNECTED = False

def ensure_connected():
    global MT5_CONNECTED
    if not MT5_CONNECTED:
        if not mt5.initialize():
            raise ConnectionError("[ERROR] Could not initialize MT5 terminal.")
        if not mt5_connect.connect():
            raise ConnectionError("[ERROR] Could not connect to MT5 server.")
        MT5_CONNECTED = True
        print("[Info] MT5 connection established successfully.")

def get_market_data(symbol, start_date, end_date, interval='1d', **kwargs):
    """
    Fetch historical market data from MetaTrader 5.

    Parameters:
        symbol (str): Symbol in MT5
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        interval (str): '1m', '5m', '15m', '1h', '1d', etc.

    Returns:
        pd.DataFrame: with standardized columns
    """

    ensure_connected()  # <=== ONLY ONCE
    #
    # # Ensure connected
    # if not mt5_connect.connect():
    #     raise ConnectionError("[ERROR] Could not connect to MT5 server to fetch data.")


    # Resolve timeframe
    timeframe = INTERVAL_MAP.get(interval)
    if timeframe is None:
        print(f"[ERROR] Unsupported interval '{interval}'. Supported intervals are: {', '.join(INTERVAL_MAP.keys())}.")
        return pd.DataFrame()

    # Convert start and end dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Fetch historical rates
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)

    if rates is None or len(rates) == 0:
        print(f"[ERROR] No data returned for symbol '{symbol}' in the specified date range.")
        return pd.DataFrame()

    # Build DataFrame
    data = pd.DataFrame(rates)

    # Convert timestamps to datetime
    data['date'] = pd.to_datetime(data['time'], unit='s')

    # Rename columns to standard
    data.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
    }, inplace=True)

    # Create 'adj_close' as same as 'close'
    data['adj_close'] = data['close']

    # Select only standard columns
    data = data[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

    # Sort by date
    data.sort_values('date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    print(f"[INFO] Successfully fetched {len(data)} rows of data for {symbol} from {start_date} to {end_date}.")

    return data
