# ./OilAnalyzer/data_fetcher.py 

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Set up a requests session with a custom header to avoid blocking
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
})

# Yahoo Finance limitations for intraday intervals
INTRADAY_LIMITS = {
    '1m': 7,
    '2m': 60,
    '5m': 60,
    '15m': 60,
    '30m': 60,
    '60m': 730,  # 2 years
    '90m': 60,
}

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


def get_market_data(symbol, start_date, end_date, interval='1d', auto_adjust=False):
    """
    Fetch historical market data for a specified symbol from Yahoo Finance.

    Parameters:
        symbol (str): Yahoo Finance ticker symbol (e.g., 'CL=F' for Crude Oil Futures).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data frequency ('1d', '1wk', '1mo', etc.). Default is '1d'.

    Returns:
        pd.DataFrame: DataFrame containing historical market data.
    """

    # Validate dates for intraday intervals
    start_date, end_date = validate_date_range(interval, start_date, end_date)

    try:
        print(f"[Info] Fetching {interval} data for {symbol} from {start_date} to {end_date}")
        
        # Fetch data from Yahoo Finance
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            session=session,
            progress=False,
            auto_adjust=auto_adjust
        )
        
    except Exception as e:
        print(f"[Error] Fetching data for {symbol}: {e}")
        return pd.DataFrame()

    if data.empty:
        print(f"[Warning] No data retrieved for {symbol}. Check symbol and date range.")
        return pd.DataFrame()

    # Reset index to turn datetime index into a column
    data.reset_index(inplace=True)

    # Rename time column consistently to 'date'
    # yfinance uses 'Date' or 'Datetime' depending on interval
    time_col = None
    for candidate in ['Date', 'Datetime']:
        if candidate in data.columns:
            time_col = candidate
            break

    if time_col is None:
        print(f"[Warning] No Date/Datetime column found. Using index as 'date'.")
        data['date'] = data.index
    else:
        data.rename(columns={time_col: 'date'}, inplace=True)

    # Ensure 'date' is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Handle MultiIndex column names (flatten if necessary)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(level) for level in col if level]) for col in data.columns]

    # Ensure all column names are lowercase and spaces replaced
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]

    # Display first few rows for verification
    print(f"[Info] Data fetched successfully for {symbol}:\n", data.head())

    # [Info]: Show dataset description
    print("\n[Info] Dataset Metadata:")
    print(f"- Timeframe: {interval}")
    print(f"- Date Range: {start_date} to {end_date}")
    print(f"- Rows: {len(data)}")
    print(f"- Columns ({len(data.columns)}): {list(data.columns)}")
    print(f"- First Date: {data['date'].iloc[0] if not data.empty else 'N/A'}")
    print(f"- Last Date: {data['date'].iloc[-1] if not data.empty else 'N/A'}")

    # Preview first and last 3 rows (optional for deeper debugging)
    print("\n[Preview] First 3 rows:\n", data.head(3))
    print("\n[Preview] Last 3 rows:\n", data.tail(3))

    return data


# End of data_fetcher.py
