# ./src/data_fetcher_agnostic.py

import pandas as pd
from src.datasources import yfinance_fetcher, mt5_fetcher

SUPPORTED_SOURCES = {
    'yfinance': yfinance_fetcher,
    'mt5': mt5_fetcher,  # dupplicated for now until MT5 is implemented
    # Future: 'csv': csv_loader, etc.
}

def get_market_data(source, symbol, start_date, end_date, interval='1d', **kwargs):
    """
    Unified interface to fetch historical market data from supported sources.
    Ensures output is in standardized OHLCV format.
    
    Parameters:
        source (str): One of 'yfinance', 'mt5', etc.
        symbol (str): Market symbol
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        interval (str): Timeframe, e.g. '1d', '1m'

    Returns:
        pd.DataFrame: With standardized columns: date, open, high, low, close, adj_close, volume
    """

    source = source.lower()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"[Error] Data source '{source}' is not supported.")

    data = SUPPORTED_SOURCES[source].get_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        **kwargs
    )

    required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"[Error] Missing required column '{col}' in data fetched from {source}.")

    return data

# End of data_fetcher_agnostic.py
