# ./src/label_candlesitcks.py
import pandas as pd


def simple_candle_state(row, atr_column='atr_14', threshold_ratio=0.3):
    """
    Classify simplified candle state as BULL, BEAR, or DOJI
    based on the body size relative to ATR.

    Parameters:
        row (pd.Series): A row of the DataFrame with OHLC and ATR values.
        atr_column (str): Name of the column with ATR value.
        threshold_ratio (float): Minimum body/ATR ratio to consider as directional.

    Returns:
        str: 'BULL', 'BEAR', or 'DOJI'
    """
    open_price = row['open']
    close_price = row['close']
    atr = row.get(atr_column, 0)

    # Sanity checks
    if atr == 0 or pd.isna(atr):
        return "DOJI"

    body_size = abs(close_price - open_price)
    body_ratio = body_size / atr

    if body_ratio < threshold_ratio:
        return "DOJI"
    elif close_price > open_price:
        return "BULL"
    else:
        return "BEAR"


def classify_candle(open_price, high_price, low_price, close_price):
    body = abs(close_price - open_price)
    total_range = high_price - low_price
    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price

    # Avoid division by zero
    if total_range == 0:
        return "Four Price Doji"

    body_ratio = body / total_range
    upper_shadow_ratio = upper_shadow / total_range
    lower_shadow_ratio = lower_shadow / total_range

    # --- Doji Variants ---
    if body_ratio < 0.05:
        if lower_shadow_ratio > 0.4 and upper_shadow_ratio < 0.05:
            return "Dragonfly Doji"
        elif upper_shadow_ratio > 0.4 and lower_shadow_ratio < 0.05:
            return "Gravestone Doji"
        elif upper_shadow_ratio > 0.4 and lower_shadow_ratio > 0.4:
            return "Long-Legged Doji"
        else:
            return "Doji"

    # --- Rickshaw Man ---
    if body_ratio < 0.05 and abs(upper_shadow_ratio - lower_shadow_ratio) < 0.1:
        return "Rickshaw Man"

    # --- Marubozu Variants ---
    if upper_shadow_ratio < 0.01 and lower_shadow_ratio < 0.01:
        if close_price > open_price:
            return "Bullish Marubozu"
        else:
            return "Bearish Marubozu"
    if upper_shadow_ratio < 0.01:
        return "Shaven Head"
    if lower_shadow_ratio < 0.01:
        return "Shaven Bottom"

    # --- Hammer / Hanging Man ---
    if lower_shadow_ratio >= 2 * body_ratio and upper_shadow_ratio < 0.1:
        if close_price > open_price:
            return "Hammer"
        else:
            return "Hanging Man"

    # --- Inverted Hammer / Shooting Star ---
    if upper_shadow_ratio >= 2 * body_ratio and lower_shadow_ratio < 0.1:
        if close_price > open_price:
            return "Inverted Hammer"
        else:
            return "Shooting Star"

    # --- Spinning Top ---
    if body_ratio >= 0.1 and body_ratio <= 0.3 and upper_shadow_ratio > 0.3 and lower_shadow_ratio > 0.3:
        if close_price > open_price:
            return "Bullish Spinning Top"
        else:
            return "Bearish Spinning Top"

    # --- High Wave Candle ---
    if upper_shadow_ratio > 0.4 and lower_shadow_ratio > 0.4 and body_ratio < 0.2:
        return "High Wave Candle"

    # --- Long / Short Day Candles ---
    if body_ratio > 0.6:
        if close_price > open_price:
            return "Bullish Long Day"
        else:
            return "Bearish Long Day"

    if body_ratio < 0.1 and upper_shadow_ratio < 0.1 and lower_shadow_ratio < 0.1:
        return "Short Day Candle"

    # --- Default Catch ---
    if close_price > open_price:
        return "Bullish Candle"
    else:
        return "Bearish Candle"



# end of label_candlesitcks.py
