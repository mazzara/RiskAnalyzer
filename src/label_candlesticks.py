# ./src/label_candlesitcks.py


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
