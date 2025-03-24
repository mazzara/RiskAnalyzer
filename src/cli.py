# ./OilAnalysis/cli.py
import pandas as pd
from datetime import datetime


# Yahoo Finance limitations for intraday intervals (in days)
YFINANCE_INTRADAY_LIMITS = {
    '1m': 7,
    '2m': 60,
    '5m': 60,
    '15m': 60,
    '30m': 60,
    '60m': 730,
    '1h': 730,
    '90m': 60
}

AVAILABLE_SYMBOLS = {
    'CL=F': 'Crude Oil Futures (NYMEX)',
    'NG=F': 'Natural Gas Futures',
    'GC=F': 'Gold Futures',
    'SI=F': 'Silver Futures',
    'ZC=F': 'Corn Futures',
    'ZS=F': 'Soybean Futures',
    'ZW=F': 'Wheat Futures'
}

def display_symbol_options():
    print("\n=== Suggested Symbols ===")
    print("\n* or any other Yahoo Finance symbol *")
    for sym, desc in AVAILABLE_SYMBOLS.items():
        print(f"{sym:<8} - {desc}")
    print("=" * 30)


def calculate_start_date(end_date, timeframe):
    """
    Dynamically determine start date based on timeframe.
    """
    if timeframe in YFINANCE_INTRADAY_LIMITS:
        # return end_date - pd.DateOffset(days=50)
        return end_date - pd.DateOffset(
            days=YFINANCE_INTRADAY_LIMITS[timeframe]-1)

    # Daily/longer timeframes defaults
    if timeframe == '1d':
        return end_date - pd.DateOffset(years=10)

    # Fallback
    return end_date - pd.DateOffset(years=5)


def prompt_user_inputs():
    """
    Prompts user for symbol, timeframe, and end date. Calculates start date automatically.
    Returns a configuration dictionary for the analysis run.
    """
    print("\n=== Oil Analyzer CLI ===")
    
    # Prompt symbol
    display_symbol_options()
    symbol = input("Enter symbol to analyze [default CL=F]: ").strip().upper() or 'CL=F'

    # Prompt timeframe
    valid_timeframes = ['1d', '1h', '1m', '5m', '15m', '30m']
    timeframe_prompt = f"Enter timeframe ({' / '.join(valid_timeframes)}) [default 1d]: "
    timeframe = input(timeframe_prompt).strip().lower() or '1d'
    if timeframe in YFINANCE_INTRADAY_LIMITS:
        max_days = YFINANCE_INTRADAY_LIMITS[timeframe]
        print(f"\n[Note] Yahoo Finance limits {timeframe} data to the last {max_days} days.")

    while timeframe not in valid_timeframes:
        print(f"Invalid timeframe! Choose from {valid_timeframes}")
        timeframe = input("Enter timeframe (1d / 1h / 1m) [default 1d]: ").strip() or '1d'

    # Prompt end date
    end_date_input = input("Enter end date (YYYY-MM-DD) [default today]: ").strip()
    
    if end_date_input == '':
        end_date = pd.to_datetime('today').normalize()
    else:
        try:
            end_date = pd.to_datetime(end_date_input)
        except Exception as e:
            print(f"Invalid date format: {e}")
            return None

    # Auto calculate start date
    start_date = calculate_start_date(end_date, timeframe)

    # Display choices
    print(f"\n[Info] Running analysis for:")
    print(f"Symbol     : {symbol}")
    print(f"Timeframe  : {timeframe}")
    print(f"Date Range : {start_date.date()} to {end_date.date()}")

    # Confirm run
    confirm = input("\nConfirm? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted by user.")
        return None

    # Return config dictionary
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }
