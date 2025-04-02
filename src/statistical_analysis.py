# ./src/statistical_analysis.py

import pandas as pd
import numpy as np

def compute_sma(df, column, period=14):
    """
    Compute Simple Moving Average (SMA) over a given period
    """
    df[f'sma_{period}'] = df[column].rolling(window=period).mean()
    return df


def compute_sma_fast(df, column, period=7):
    """
    Compute Fast Simple Moving Average (SMA) over a given period
    """
    df[f'sma_fast'] = df[column].rolling(window=period).mean()
    return df


def compute_sma_slow(df, column, period=21):
    """
    Compute Slow Simple Moving Average (SMA) over a given period
    """
    df[f'sma_slow'] = df[column].rolling(window=period).mean()
    return df


def compute_daily_return(df):
    """
    Compute Daily Return (%)
    """
    df['daily_return_%'] = df['close'].pct_change() * 100
    df['daily_return_decimal'] = df['close'].pct_change()
    return df

def compute_daily_range(df):
    """
    Compute Daily Range (%) = (High - Low) / Low * 100
    """
    df['daily_range_%'] = (df['high'] - df['low']) / df['low'] * 100
    df['daily_range_decimal'] = (df['high'] - df['low']) / df['low']
    return df

def compute_true_range(df):
    """
    Compute True Range (TR)
    TR_t = max(High - Low, |High - Prev_Close|, |Low - Prev_Close|)
    """
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df.apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['prev_close']),
            abs(row['low'] - row['prev_close'])
        ),
        axis=1
    )
    return df


def compute_atr(df, period=14):
    """
    Compute Average True Range (ATR) over a given period
    """
    if 'tr' not in df.columns:
        df = compute_true_range(df)
    
    df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()
    return df


def compute_gain_loss_chains(df, return_column='daily_return_%', threshold=0.0):
    """
    Compute Gain and Loss Chains (bullish/bearish streaks)
    """
    # 1 for gain, -1 for loss, 0 for no change
    df['gain_loss'] = np.sign(df['daily_return_%'].fillna(0))

    # Initialize counters
    bullish_chain = []
    bearish_chain = []
    bull_bear_chain = []

    bull_count = 0
    bear_count = 0

    for change in df['gain_loss']:
        if change > 0:
            bull_count += 1
            bear_count = 0
        elif change < 0:
            bear_count += 1
            bull_count = 0
        else:
            bull_count = 0
            bear_count = 0

        bullish_chain.append(bull_count)
        bearish_chain.append(bear_count)

        if bull_count > 0:
            bull_bear_chain.append(bull_count)
        elif bear_count > 0:
            bull_bear_chain.append(-bear_count)
        else:
            bull_bear_chain.append(0)

    df['bullish_chain'] = bullish_chain
    df['bearish_chain'] = bearish_chain
    df['bull_bear_chain'] = bull_bear_chain

    return df

def run_statistical_analysis(df):
    """
    Run full statistical feature engineering pipeline
    """
    print("[Info] Starting statistical feature engineering...")
    
    df = compute_daily_return(df)
    df = compute_daily_range(df)
    df = compute_true_range(df)
    df = compute_atr(df, period=14)
    df = compute_gain_loss_chains(df)
    df = compute_sma_fast(df, 'close', period=7)
    df = compute_sma_slow(df, 'close', period=21)
    
    # Drop temp columns
    if 'prev_close' in df.columns:
        df.drop(columns=['prev_close'], inplace=True)
    
    print("[Info] Statistical feature engineering completed!")
    return df

