# ./RiskAnalyzer/main.py  
from src.cli import prompt_user_inputs
from src.data_fetcher import get_market_data as get_yfinance_data 
from src.data_fetcher_agnostic import get_market_data as get_agnostic_data
from src.statistical_analysis import run_statistical_analysis
from src.filters import adaptive_outlier_filter
from src.descriptive_analysis import descriptive_statistics_pipeline
from src.distribution_analysis import distribution_analysis_pipeline
from src.conditional_probabilities import conditional_probabilities_pipeline, full_conditional_probabilities, classify_state_dynamic, get_dynamic_neutral_threshold
from src.volatility_analysis import fit_garch, plot_volatility, forecast_volatility
from src.volatility_regimes import hmm_volatility_pipeline
from src.monte_carlo import monte_carlo_simulation, plot_simulations, calculate_var_cvar
from src.monte_carlo_v2 import monte_carlo_simulation_v2, plot_simulations, calculate_var_cvar
from src.trading_rules import determine_regime_strategy, determine_trade_signal
from src.label_candlesticks import classify_candle, simple_candle_state
from src.report_generator import generate_report, full_conditional_probability_lookup_full, full_conditional_probability_lookup_full_verbose
from src.logger_config import logger  
import argparse
import json 
import numpy as np


# CLI argument parser
parser = argparse.ArgumentParser(description="Run RiskAnalyzer in normal or scanner mode.")
parser.add_argument('--debug', action='store_true', help='Run in step-by-step debug mode')
parser.add_argument('--scanner', action='store_true', help='Run scanner mode on predefined symbols')
parser.add_argument('--positions', action='store_true', help='Analyze positions from config/positions.json')
parser.add_argument('--no-prompt', action='store_true', help='Skip CLI prompts and use config.json only')
parser.add_argument('--risk-explorer', action='store_true', help='Run risk explorer mode')
args = parser.parse_args()

with open("config/config.json") as f:
    settings = json.load(f)
debug_mode = settings.get("debug_mode", False)

debug_mode = args.debug or settings.get("debug_mode", False)


def pause_step(step_name):
    if debug_mode:
        input(f"\n[Paused] Step: {step_name} complete. Press Enter to continue...")

# Function to print terminal verbose if whenever in debug mode 
def print_verbose(message, tag=None):
    if not debug_mode:
        return
    prefix = f"[{tag}] " if tag else "[Verbose] "
    if isinstance(message, str):
        print(f"{prefix} {message}")
    elif hasattr(message, "to_string"):  # For DataFrames or Series
        print(f"{prefix}\n{message.to_string(index=False)}")
    else:
        print(f"{prefix} {message}")



def run_analysis(config, position=None):
    symbol = config['symbol']
    start = config['start_date']
    end = config['end_date']
    interval = config['timeframe']

    # Safely extract symbol-based settings
    symbol_meta = settings.get("symbol", {}).get(symbol, settings["symbol"]["default"])
    contract_size = symbol_meta.get("contract_size", 1)

    # Initialize remaining config
    threshold = settings["conditional_probability_state_threshold"]
    n_steps = settings["monte_carlo"]["n_steps"]
    return_column = settings["return_column"]
    outlier_method = settings["outlier_filtering"]["method"]
    z_thresh = settings["outlier_filtering"]["z_thresh"]
    lower_quantile = settings["outlier_filtering"]["lower_quantile"]
    upper_quantile = settings["outlier_filtering"]["upper_quantile"]
    audit_filename = settings["outlier_filtering"]["audit_filename"]
    show_plots = settings["show_plots"]
    conditional_probability_state_threshold = settings["conditional_probability_state_threshold"]
    max_chain_length = settings["max_chain_length"]
    state_column = settings["state_column"]
    hmm_n_states = settings["hmm_n_states"]
    # Load dynamic threshold tuning params
    state_config = settings.get("state_classification", {}).get("default", {})
    symbol_config = settings.get("state_classification", {}).get(symbol, {})
    time_config = symbol_config.get(interval, {})
    risk_per_trade = settings.get("risk_per_trade", 0.01)
    account_size = settings.get("account_size", 50000)

    std_cap_multiplier = time_config.get("std_cap_multiplier",
                         state_config.get("std_cap_multiplier", 0.4))

    intraday_bias = time_config.get("intraday_bias",
                    state_config.get("intraday_bias", 0.6 if 'm' in interval or 'h' in interval else 1.0))

    # print(f"\n[Info] Fetching data for {symbol} from {start} to {end}, timeframe: {interval}")
    verbose = f"Symbol: {symbol}, Start: {start}, End: {end}, Timeframe: {interval}"
    print_verbose(verbose)


    # ==== Step 1: Fetch data ================================
    # raw_data = get_market_data(symbol, start, end, interval)
    # if raw_data.empty:
    #     print("[Error] No data fetched. Exiting.")
    #     raise ValueError(f"No data fetched for {symbol}")
    # pause_step("Data Fetching")

    # New data fetching agnostic
    source = config.get("data_source") or settings.get("data_source", "yfinance")
    # raw_data = get_agnostic_data(source, symbol, start, end, interval)
    try:
        raw_data = get_agnostic_data(source, symbol, start, end, interval)
    except ValueError as e:
        logger.error(f"[1833:30:30 Error] Data fetching failed: {e}")
        raise

    if raw_data is None or raw_data.empty:
        print("[Error] No data fetched. Exiting.")
        raise ValueError(f"No data fetched for {symbol}")

    pause_step("Data Fetching")

    # ==== Step 2: Run statistical analysis
    raw_data = run_statistical_analysis(raw_data)
    raw_data.to_csv('data_raw.csv', index=False)
    pause_step("Statistical Analysis - Raw Data (Pre-Filtering)")

    # Step 2.1: Filter outliers using combined method (Z-score + Quantile)
    data_filtered = adaptive_outlier_filter(
        raw_data,
        column=return_column,
        method=outlier_method,
        z_thresh=z_thresh,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        audit_filename='data_excluded_outliers.csv'
    )
    data_filtered.to_csv('data_filtered.csv', index=False)

    data = data_filtered.copy()

    pause_step("Statistical Analysis & Outlier Filtering")

    # ==== Step 3: Descriptive stats & distribution analysis
    columns_to_analyze = ['daily_return_%', 'daily_range_%', 'tr', 'atr_14']

    # Capture Returns stats
    # Function returns a dictionary to be used in the report
    desc_stats = descriptive_statistics_pipeline(data, columns_to_analyze, show_plots=show_plots)
    dist_results = distribution_analysis_pipeline(data, column=return_column, thresholds=[1,2,3], show_plots=show_plots)
    pause_step("Distribution Analysis")

    # ==== Step 4: Label Candlesticks
    data['candlestick'] = data.apply(
        lambda row: classify_candle(row['open'], row['high'], row['low'], row['close']),
        axis=1
    )
    data['candle_state'] = data.apply(simple_candle_state, axis=1)
    pause_step("Candlestick Labeling")

    # ==== Step 5: Conditional Probabilities & Continuation Patterns
    cond_probs_results = conditional_probabilities_pipeline(data,
                                                            threshold=conditional_probability_state_threshold,
                                                            max_chain=max_chain_length,
                                                            state_column=state_column,
                                                            show_plots=show_plots)
    pause_step("Conditional Probabilities & Continuation Patterns")

    # NEW: Full Conditional Probabilities for All States
    full_cond_probs = full_conditional_probabilities(data, state_column=state_column, max_chain=max_chain_length)

    # Debugging: Print example for chain length 5
    # print(full_cond_probs.keys())
    # print("[Debug] Example for chain length 5:", full_cond_probs.get(5, {}))

    # After conditional_probs_results is generated...
    # print("\n[Verbose] Conditional Probability Chain Analysis...")
    verbose = f"Conditional Probability Chain Analysis for '{state_column}'"
    print_verbose(verbose)

    # Current chain snapshot (up to max_chain)
    max_chain_len = max(full_cond_probs.keys(), default=5)
    current_chain = tuple(data[state_column].iloc[-max_chain_len:])

    # print(f"[Verbose] Current Chain (latest {max_chain_len} states): {' > '.join(current_chain)}")
    # print(f"[Verbose] Current Chain (latest {max_chain_len} states): {' > '.join(map(str, current_chain))}")
    verbose = f"[Verbose] Current Chain (latest {max_chain_len} states): {' > '.join(map(str, current_chain))}"
    print_verbose(verbose)


    # Run the lookup function (reusing the function from report_generator.py)
    probability, next_state_probs = full_conditional_probability_lookup_full_verbose(data, full_cond_probs, state_column)

    # Display results
    # print(f"\n[Verbose] Continuation Probability for Chain '{' > '.join(current_chain)}':")
    # print(f"\n[Verbose] Continuation Probability for Chain '{' > '.join(map(str, current_chain))}':")
    verbose = f"Continuation Probability for Chain '{' > '.join(map(str, current_chain))}':"
    print_verbose(verbose)

    if probability is not None:
        # print(f"- Continuation Probability (stay {current_chain[-1]}): {probability * 100:.2f}%")
        verbose = f"Continuation Probability (stay {current_chain[-1]}): {probability * 100:.2f}%"
        print_verbose(verbose)
    else:
        print("[Warning] - No continuation probability available for this chain.")

    # print("\n[Verbose] Probabilities for Next State (Chain Lookup):")
    verbose = "Probabilities for Next State (Chain Lookup):"
    print_verbose(verbose)
    if next_state_probs:
        for state, prob in next_state_probs.items():
            direction = "continuation" if state == current_chain[-1] else "reversal"
            # print(f"- {state} ({direction}): {prob * 100:.2f}%")
            verbose = f"{state} ({direction}): {prob * 100:.2f}%"
            print_verbose(verbose)
    else:
        print("[Warning] - No next state probabilities available for this chain.")

    pause_step("Full Conditional Probabilities")

    # ==== Step 6: Volatility Analysis
    garch_results = fit_garch(data, return_column=return_column)
    # plot_volatility(data, garch_results)
    forecast_df = forecast_volatility(garch_results, steps=5)
    print_verbose(forecast_df)
    # print(forecast_df)
    pause_step("Volatility Analysis")

    # ==== Step 7: Run HMM regime detection
    hmm_model, data_with_regimes = hmm_volatility_pipeline(data, return_column=return_column, n_states=hmm_n_states, show_plots=False)
    # low_vol_state = np.argmin(hmm_model.covars_.flatten())
    # low_vol_std = np.sqrt(hmm_model.covars_.flatten()[low_vol_state])
    # dynamic_threshold = low_vol_std * 0.01

    # Base threshold from HMM
    neutral_threshold = get_dynamic_neutral_threshold(hmm_model) * 100

    # Cap using return std
    std_cap = data['daily_return_%'].std() * std_cap_multiplier

    # Final threshold (capped and adjusted)
    neutral_threshold = min(neutral_threshold, std_cap)
    neutral_threshold *= intraday_bias


    # Get dynamic threshold and convert to %
    # neutral_threshold = get_dynamic_neutral_threshold(hmm_model) * 100
    #
    # intraday_bias = 0.5 if 'm' in interval or 'h' in interval else 1.0
    # neutral_threshold *= intraday_bias

    # Classify using dynamic threshold
    data['state'] = data.apply(lambda row: classify_state_dynamic(row, threshold=neutral_threshold), axis=1)

    print(f"[Info] Dynamic neutral threshold set to ±{neutral_threshold:.4f}%")
    print("[Info] State Distribution:")
    print(data['state'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))


    pause_step("HMM Regime Detection")

    # ==== Step 8: Monte Carlo Simultion - v1.0 (Simple)
    # Get current price
    current_price = data['close'].iloc[-1]
    # Simple mu/sigma estimates from historical data
    mu = data['daily_return_%'].mean()/100
    sigma = data['daily_return_%'].std()/100
    # print(f"Current Prices: {current_price}, Mu: {mu:.5f}, Sigma: {sigma:.5f}")
    verbose = f"Current Prices: {current_price}, Mu: {mu:.5f}, Sigma: {sigma:.5f}"
    print_verbose(verbose)

    n_steps = 10
    n_simulations = 500

    simulated_price_paths = monte_carlo_simulation(
            start_price=current_price,
            mu=mu,
            sigma=sigma,
            n_steps=n_steps,
            n_simulations=n_simulations)
    # Plot simulated price paths
    # plot_simulations(simulated_price_paths)
    # Calculate VaR and CVaR
    var_v1, cvar_v1 = calculate_var_cvar(simulated_price_paths, confidence_level=0.95)


    # ==== Step 9: Monte carlo v2.0 (Regime + Volatility Aware)
    current_price = data['close'].iloc[-1]
    print(f"Current Price: {current_price}")
    verbose = f"Current Price: {current_price}"
    print_verbose(verbose)

    n_steps = 10
    n_simulations = 500

    simulated_price_paths = monte_carlo_simulation_v2(
        start_price=current_price,
        regimes_df=data_with_regimes,
        garch_results=garch_results,
        hmm_model=hmm_model,
        n_steps=n_steps,
        n_simulations=n_simulations
    )

    # Plot simulations
    # plot_simulations(simulated_price_paths)

    # Risk metrics
    var_v2, cvar_v2 = calculate_var_cvar(simulated_price_paths, confidence_level=0.95)

    pause_step("Monte Carlo Simulations")


    # ==== Step 10: Actionable Trading Insights
    current_regime = data_with_regimes['hmm_state'].iloc[-1]
    next_vol_forecast = forecast_volatility(garch_results, steps=1).iloc[0]['forecast_volatility']
    continuation_prob = 0.65  # Example from Markov or Chains
    reversal_prob = 0.35      # Example from Markov or Chains
    current_price = data['close'].iloc[-1]
    mean_price = data['close'].rolling(window=20).mean().iloc[-1]  # 20-SMA

    # Generate trade signal
    trade_signal = determine_trade_signal(
        regime=current_regime,
        current_price=current_price,
        mean_price=mean_price,
        contract_size=contract_size,
        continuation_prob=continuation_prob,
        reversal_prob=reversal_prob,
        garch_vol=next_vol_forecast,
        account_size=account_size,
        risk_per_trade=risk_per_trade
    )

    # Display output
    # print("\n[Trade Signal]")
    verbose = "\n[Trade Signal]"
    print_verbose(verbose)
    for k, v in trade_signal.items():
        # print(f"{k}: {v}")
        verbose = f"{k}: {v}"
        print_verbose(verbose)

    # Show and/or save
    # print(data.tail(20))
    verbose = data.tail(20)
    print_verbose(verbose)
    data.to_csv('data_with_stats.csv', index=False)

    pause_step("Trading Insights")

    # Final step === Generate Report ===
    generate_report(
        symbol=symbol,
        timeframe=interval,
        start_date=start,
        end_date=end,
        data=data,
        descriptive_stats=desc_stats,
        distribution_results=dist_results,
        conditional_probs=cond_probs_results,
        full_cond_probs=full_cond_probs,
        garch_results=garch_results,  # fine, keep if you want HMM link
        volatility_forecast=forecast_df,  # <-- pass this explicitly!
        hmm_results=hmm_model,
        monte_carlo={
            'v1': {'var': var_v1, 'cvar': cvar_v1},
            'v2': {'var': var_v2, 'cvar': cvar_v2}
        },
        trade_signal=trade_signal,
        state_column=state_column
    )

    # Print tail of data
    # print(data.tail())
    verbose = data.tail()
    print_verbose(verbose)

    print("\n[Info] Analysis Complete.\n")

    if position:
        position_size = position.get("position_size", 1.0)
        entry_price = position.get("entry_price")
        side = position.get("side", "long").lower()
        stop_loss = trade_signal.get("stop_loss")
        take_profit = trade_signal.get("take_profit")
        market_value = position_size * current_price
        unrealized_pnl = (current_price - entry_price) * position_size if side == "long" else (entry_price - current_price) * position_size
        pnl_percent = (unrealized_pnl / (entry_price * position_size)) * 100 if entry_price and position_size else 0

        # Tactical Stop-based Risk
        if stop_loss:
            if side == "long":
                stop_risk = max((entry_price - stop_loss) * position_size, 0)
            else:
                stop_risk = max((stop_loss - entry_price) * position_size, 0)
        else:
            stop_risk = None

        evaluation = {
            "symbol": symbol,
            "side": side,
            "position_size": position_size,
            "entry_price": entry_price,
            "entry_date": position.get("entry_date"),
            "current_price": current_price,
            "market_value": market_value,
            "volatility_regime": trade_signal.get("regime"),
            "strategy_bias": trade_signal.get("bias"),
            "environment_action": trade_signal.get("trade_action"),
            "confidence": trade_signal.get("confidence"),
            "stop_loss": stop_loss,
            "pnl": unrealized_pnl,
            "pnl_percent": pnl_percent,
            "take_profit": take_profit,
            "stop_risk": stop_risk,
            "var_95": var_v2 * market_value,
            "cvar_95": cvar_v2 * market_value,
            "recommended_exit": False  # Placeholder logic
        } 
        return evaluation

    return trade_signal





if __name__ == '__main__':
    if args.scanner:
        from src.scanner import scanner_mode
        scanner_mode()
        exit()
    
    if args.positions:
        from src.analyze_positions import analyze_positions
        analyze_positions()
        exit()

    # # CLI interaction
    # config = prompt_user_inputs()
    #
    # if config is None:
    #     exit()
    # run_analysis(config)

    if args.risk_explorer:
        from src.risk_explorer import risk_explorer_mode
        risk_explorer_mode()
        exit()

    if args.no_prompt:
        # Skip user prompts, use config.json settings
        config = {
            "symbol": settings.get("run_symbol",{}).get("symbol", "CL=F"),
            "timeframe": settings.get("run_symbol",{}).get("default_timeframe", "1d"),
            "start_date": settings.get("run_symbol",{}).get("default_start_date", "2015-01-01"),
            "end_date": settings.get("run_symbol",{}).get("default_end_date", "2025-04-21")
            # "source": settings.get("run_symbol",{}).get("source", "yfinance")
        }

        print("\n[Info] Running RiskAnalyzer with config from config.json:")
        print(f"  Symbol    : {config['symbol']}")
        print(f"  Timeframe : {config['timeframe']}")
        print(f"  Start Date: {config['start_date']}")
        print(f"  End Date  : {config['end_date']}")
        print("")

        run_analysis(config)
    else:
        # Normal interactive mode
        config = prompt_user_inputs()
        if config is None:
            exit()
        run_analysis(config)



# end of main.py
