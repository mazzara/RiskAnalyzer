# ./src/trading_rules.py

def determine_regime_strategy(regime, garch_volatility, chain_probs, current_price):
    """
    Suggests trading strategy based on regime, volatility, and probabilities.
    
    Args:
        regime: HMM detected state (low or high volatility)
        garch_volatility: forecast volatility from GARCH (next period)
        chain_probs: Markov/conditional probabilities (continuation/reversal)
        current_price: latest price

    Returns:
        strategy: dict with suggested action and risk parameters
    """
    strategy = {}
    
    # Define basic strategies by regime
    if regime == 0:
        strategy['regime'] = "Low Volatility"
        strategy['bias'] = "Mean Reversion"
    elif regime == 1:
        strategy['regime'] = "High Volatility"
        strategy['bias'] = "Momentum / Breakout"
    else:
        strategy['regime'] = "Unknown"
        strategy['bias'] = "Neutral"

    # Volatility-based position size (simple example)
    risk_per_trade = 0.01  # 1% account risk
    volatility_factor = garch_volatility * current_price
    position_size = risk_per_trade / volatility_factor if volatility_factor != 0 else 0
    
    strategy['position_size_factor'] = round(position_size, 4)

    # Conditional probabilities
    continuation_prob = chain_probs.get('continuation_prob', 0.5)
    reversal_prob = chain_probs.get('reversal_prob', 0.5)

    if continuation_prob > reversal_prob:
        strategy['entry_signal'] = "Follow Trend"
    else:
        strategy['entry_signal'] = "Reversal Setup"

    # Suggested stop loss and take profit
    strategy['stop_loss_distance'] = round(garch_volatility * 2 * current_price, 4)
    strategy['take_profit_distance'] = round(garch_volatility * 3 * current_price, 4)

    return strategy


def determine_trade_signal(regime, current_price, mean_price, continuation_prob, reversal_prob, garch_vol, account_size=100000, risk_per_trade=0.01):
    """
    Generate trade suggestions based on regime, volatility, probabilities, and price levels.

    Args:
        regime (int): HMM regime classification (0 or 1)
        current_price (float): latest close price
        mean_price (float): moving average (20-SMA or VWAP)
        continuation_prob (float): probability of trend continuation
        reversal_prob (float): probability of reversal
        garch_vol (float): forecasted volatility (as decimal)
        account_size (float): total account equity
        risk_per_trade (float): risk per trade (percentage of account size)

    Returns:
        dict: trade signal with parameters
    """

    strategy = {}

    # --- Regime Logic ---
    if regime == 0:
        strategy['regime'] = "Low Volatility"
        strategy['bias'] = "Mean Reversion"
    elif regime == 1:
        strategy['regime'] = "High Volatility"
        strategy['bias'] = "Momentum / Breakout"
    else:
        strategy['regime'] = "Unknown"
        strategy['bias'] = "Neutral"

    # --- Determine Trade Direction ---
    # Distance from mean price to current price
    distance_to_mean = current_price - mean_price

    # Default to HOLD, override below if conditions trigger
    entry_type = "HOLD"
    trade_action = None
    entry_price = None
    confidence = "Low"

    # Low Volatility (Mean Reversion Logic)
    if strategy['regime'] == "Low Volatility":
        if distance_to_mean > garch_vol * current_price:
            trade_action = "SELL"
            entry_price = current_price
            confidence = "Medium" if reversal_prob > 0.5 else "Low"
        elif distance_to_mean < -garch_vol * current_price:
            trade_action = "BUY"
            entry_price = current_price
            confidence = "Medium" if reversal_prob > 0.5 else "Low"
        else:
            trade_action = "HOLD"
            confidence = "Low"

    # High Volatility (Trend Following Logic)
    elif strategy['regime'] == "High Volatility":
        if continuation_prob > 0.6:
            if distance_to_mean > 0:
                trade_action = "BUY"
            else:
                trade_action = "SELL"
            entry_price = current_price
            confidence = "High"
        else:
            trade_action = "HOLD"
            confidence = "Low"

    # --- Stop-Loss & Take-Profit Calculation ---
    stop_loss_distance = garch_vol * 2 * current_price
    take_profit_distance = garch_vol * 3 * current_price

    if trade_action == "BUY":
        stop_loss_price = entry_price - stop_loss_distance
        take_profit_price = entry_price + take_profit_distance
    elif trade_action == "SELL":
        stop_loss_price = entry_price + stop_loss_distance
        take_profit_price = entry_price - take_profit_distance
    else:
        stop_loss_price = None
        take_profit_price = None

    # --- Position Sizing ---
    volatility_dollar_move = garch_vol * current_price
    if volatility_dollar_move != 0:
        position_size_factor = risk_per_trade * account_size / (2 * volatility_dollar_move)
    else:
        position_size_factor = 0

    # --- Final Structured Signal ---
    trade_signal = {
        "symbol": "CL=F",
        "regime": strategy['regime'],
        "bias": strategy['bias'],
        "trade_action": trade_action,
        "entry_type": "Market" if trade_action != "HOLD" else None,
        "entry_price": entry_price,
        "stop_loss_price": round(stop_loss_price, 4) if stop_loss_price else None,
        "take_profit_price": round(take_profit_price, 4) if take_profit_price else None,
        "stop_loss_distance": round(stop_loss_distance, 4),
        "take_profit_distance": round(take_profit_distance, 4),
        "position_size": round(position_size_factor, 2),
        "confidence": confidence
    }

    return trade_signal

# end of trading_rules.py
