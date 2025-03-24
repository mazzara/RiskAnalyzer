# ./OilAnalyzer/statistical_analysis.py

import pandas as pd
import numpy as np

def compute_daily_return(df):
    """
    Compute Daily Return (%)
    """
    df['daily_return_%'] = df['close'].pct_change() * 100
    return df

def compute_daily_range(df):
    """
    Compute Daily Range (%) = (High - Low) / Low * 100
    """
    df['daily_range_%'] = (df['high'] - df['low']) / df['low'] * 100
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

def compute_gain_loss_chains(df):
    """
    Compute Gain and Loss Chains (bullish/bearish streaks)
    """
    # 1 for gain, -1 for loss, 0 for no change
    df['gain_loss'] = np.sign(df['daily_return_%'].fillna(0))
    
    # Initialize counters
    bullish_chain = []
    bearish_chain = []
    
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
    
    df['bullish_chain'] = bullish_chain
    df['bearish_chain'] = bearish_chain
    
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
    
    # Drop temp columns
    if 'prev_close' in df.columns:
        df.drop(columns=['prev_close'], inplace=True)
    
    print("[Info] Statistical feature engineering completed!")
    return df

