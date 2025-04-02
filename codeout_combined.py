

# === FILE: codeout.py ===

# ./codeout.py 
# Run this script to iterate over all python files in the project and compile a single file with all the code snippets. 
# The output file will be saved in the same directory as this script. 
# At the end you have a single file to provide to GPT and friends. 

import os
import re


# Settings
PROJECT_DIR = "."  # Change this if running from another folder
OUTPUT_FILE = "codeout_combined.py"
EXCLUDE_DIRS = {'__pycache__', '.git', 'venv', 'env', '.mypy_cache', 'build', 'dist', 'reports'}

def collect_python_files(base_dir):
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return sorted(python_files)

def extract_code(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Failed to read {file_path}: {e}\n"

def build_combined_file(file_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        for path in file_list:
            rel_path = os.path.relpath(path, PROJECT_DIR)
            out.write(f"\n\n# === FILE: {rel_path} ===\n\n")
            code = extract_code(path)
            out.write(code)

if __name__ == "__main__":
    print(f"[Info] Collecting Python files in '{PROJECT_DIR}'...")
    all_py_files = collect_python_files(PROJECT_DIR)
    print(f"[Info] {len(all_py_files)} files found.")

    print(f"[Info] Writing combined file to '{OUTPUT_FILE}'...")
    build_combined_file(all_py_files, OUTPUT_FILE)
    print("[Success] Combined code output generated.")


# === FILE: main.py ===

# ./RiskAnalyzer/main.py  
from src.cli import prompt_user_inputs
from src.data_fetcher import get_market_data 
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
import argparse
import json 
import numpy as np

# with open("config/config.json") as f:
#     settings = json.load(f)
# debug_mode = settings.get("debug_mode", False)
#

# CLI argument parser
parser = argparse.ArgumentParser(description="Run RiskAnalyzer with optional debug mode.")
parser.add_argument('--debug', action='store_true', help='Run in step-by-step debug mode')
args = parser.parse_args()

with open("config/config.json") as f:
    settings = json.load(f)
debug_mode = settings.get("debug_mode", False)

# debug_mode = args.debug
# if args.debug:
#     debug_mode = True

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


if __name__ == '__main__':

    # CLI interaction
    config = prompt_user_inputs()

    if config is None:
        exit()

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

    # print(f"\n[Info] Fetching data for {symbol} from {start} to {end}, timeframe: {interval}")
    verbose = f"Symbol: {symbol}, Start: {start}, End: {end}, Timeframe: {interval}"
    print_verbose(verbose)


    # ==== Step 1: Fetch data
    raw_data = get_market_data(symbol, start, end, interval)
    if raw_data.empty:
        print("[Error] No data fetched. Exiting.")
        exit()
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
    std_cap = data['daily_return_%'].std() * 0.4  # 40% of return std, tweakable

    # Timeframe bias
    intraday_bias = 0.6 if 'm' in interval or 'h' in interval else 1.0

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
        account_size=50000,
        risk_per_trade=0.01
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


# end of main.py


# === FILE: src/__init__.py ===



# === FILE: src/cli.py ===

# ./src/cli.py
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
    'ZW=F': 'Wheat Futures',
    'LE=F': 'Live Cattle Futures',
    'BTC-USD': 'Bitcoin/USD',
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
    print("\n=== Risk Analyzer CLI ===")
    
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


# === FILE: src/conditional_probabilities.py ===

# ./src/conditional_probabilities.py

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Suppress warnings for cleaner outputs
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. CLASSIFY STATES
# -------------------------------
def classify_state(row, threshold=0.0):
    """
    Classify price movement state: Bullish, Bearish, or Neutral
    Args:
        row: pandas Series (single row with daily return)
        threshold: return threshold to classify as Neutral
    Returns:
        String: 'Bullish', 'Bearish', or 'Neutral'
    """
    if row['daily_return_%'] > threshold:
        return 'Bullish'
    elif row['daily_return_%'] < -threshold:
        return 'Bearish'
    else:
        return 'Neutral'


def classify_state_dynamic(row, threshold):
    if row['daily_return_%'] > threshold:
        return 'Bullish'
    elif row['daily_return_%'] < -threshold:
        return 'Bearish'
    else:
        return 'Neutral'


def get_dynamic_neutral_threshold(hmm_model):
    """
    Compute a dynamic threshold based on HMM state means.
    Returns:
        threshold: float (in decimal, e.g., 0.0042 for 0.42%)
    """
    hmm_means = hmm_model.means_.flatten()
    return max(abs(hmm_means.min()), abs(hmm_means.max())) / 2


# -------------------------------
# 2. RUNS TEST
# -------------------------------
def runs_test(df, state_column='state'):
    """
    Perform a runs test to check for randomness (trendiness vs mean reversion)
    Args:
        df: DataFrame with a 'state' column
        state_column: column name with classified states (Bullish/Bearish/Neutral)
    Returns:
        runs_info: dict with run counts and expected/observed comparisons
    """
    # Convert states to binary (Bullish=1, Bearish=0), ignore Neutral for now
    binary_seq = df[df[state_column] != 'Neutral'][state_column].map({'Bullish': 1, 'Bearish': 0}).values

    n1 = np.sum(binary_seq)
    n2 = len(binary_seq) - n1
    total_runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1])

    # Expected runs and standard deviation (Wald–Wolfowitz runs test)
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                       (((n1 + n2) ** 2) * (n1 + n2 - 1)))

    z_score = (total_runs - expected_runs) / std_runs

    print(f"\n[Runs Test]")
    print(f"Number of Bullish: {n1}")
    print(f"Number of Bearish: {n2}")
    print(f"Total Runs: {total_runs}")
    print(f"Expected Runs: {expected_runs:.2f}")
    print(f"Z-score: {z_score:.2f}")

    runs_info = {
        'bullish_count': n1,
        'bearish_count': n2,
        'total_runs': total_runs,
        'expected_runs': expected_runs,
        'z_score': z_score
    }

    return runs_info

# -------------------------------
# 3. CONDITIONAL PROBABILITY DISTRIBUTIONS
# -------------------------------
def conditional_probabilities(df, state_column='state', max_chain=5):
    """
    Compute conditional probabilities of continuation/reversal given N previous states
    Args:
        df: DataFrame with a 'state' column
        max_chain: maximum chain length to consider
    Returns:
        cond_probs: dict of chain length probabilities
    """
    cond_probs = {}
    states = df[state_column].values
    length = len(states)

    for chain_len in range(1, max_chain + 1):
        continuation_counts = Counter()
        total_counts = Counter()

        for i in range(chain_len, length - 1):
            prev_chain = tuple(states[i - chain_len:i])
            next_state = states[i]

            total_counts[prev_chain] += 1
            if prev_chain[-1] == next_state:
                continuation_counts[prev_chain] += 1

        probabilities = {
            chain: continuation_counts[chain] / total_counts[chain]
            for chain in total_counts
        }

        cond_probs[chain_len] = probabilities

        # Print conditional probabilities -- mostly for debugging
        # print(f"\n[Conditional Probabilities] Chain length: {chain_len}")
        # for chain, prob in probabilities.items():
        #     print(f"Prev {chain} => Continuation Probability: {prob:.2f}")

    return cond_probs

# -------------------------------
# 4. MARKOV CHAIN ANALYSIS (TPM)
# -------------------------------
def markov_chain_tpm(df, state_column='state'):
    """
    Calculate Markov Chain Transition Probability Matrix (TPM)
    Args:
        df: DataFrame with a 'state' column
    Returns:
        tpm_df: DataFrame TPM
    """
    states = df[state_column].values
    # unique_states = ['Bullish', 'Bearish', 'Neutral']
    unique_states = df[state_column].dropna().unique().tolist()

    transitions = {state: Counter() for state in unique_states}

    for current, next_ in zip(states[:-1], states[1:]):
        transitions[current][next_] += 1

    tpm = {}
    for state in unique_states:
        total = sum(transitions[state].values())
        tpm[state] = {next_state: (transitions[state][next_state] / total) if total > 0 else 0
                      for next_state in unique_states}

    tpm_df = pd.DataFrame(tpm).T
    print("\n[Transition Probability Matrix (TPM)]")
    print(tpm_df)

    return tpm_df

# -------------------------------
# 5. TPM HEATMAP VISUALIZATION
# -------------------------------
def plot_tpm_heatmap(tpm_df):
    """
    Plot Transition Probability Matrix as a heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(tpm_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title("Transition Probability Matrix (TPM) Heatmap")
    plt.ylabel("Current State")
    plt.xlabel("Next State")
    plt.show()

# -------------------------------
# 6. PIPELINE FUNCTION
# # -------------------------------

def conditional_probabilities_pipeline(df, threshold=0.0, max_chain=5, show_plots=True, state_column='state'):
    print("\n[Info] Starting Conditional Probabilities & Continuation Patterns Analysis...")

    if state_column == 'state':  # Only classify if using default column
        df[state_column] = df.apply(lambda row: classify_state(row, threshold=threshold), axis=1)

    # No filtering unless explicitly needed
    # df = df[df[state_column] != "Neutral"].copy()

    runs_info = {}
    if df[state_column].nunique() == 2:
        runs_info = runs_test(df, state_column=state_column)
    else:
        print(f"[Warning] Runs test skipped: '{state_column}' has {df[state_column].nunique()} unique values (expected binary).")

    cond_probs = conditional_probabilities(df, state_column=state_column, max_chain=max_chain)
    tpm_df = markov_chain_tpm(df, state_column=state_column)

    if show_plots:
        plot_tpm_heatmap(tpm_df)

    return {
        'runs_info': runs_info,
        'cond_probs': cond_probs,
        'tpm': tpm_df
    }


def full_conditional_probabilities(df, state_column='state', max_chain=5):
    """
    Compute full conditional probabilities for all next states 
    given N previous states (chain of length N).

    Returns:
        dict of: {chain_len: {prev_chain: {state: prob}}}
    """
    from collections import Counter

    full_cond_probs = {}
    states = df[state_column].values
    length = len(states)

    for chain_len in range(1, max_chain + 1):
        next_state_counts = {}  # holds {prev_chain: {next_state: count}}
        total_counts = {}

        for i in range(chain_len, length - 1):
            prev_chain = tuple(states[i - chain_len:i])
            next_state = states[i]

            # Initialize if new chain
            if prev_chain not in next_state_counts:
                next_state_counts[prev_chain] = Counter()

            # Count transitions
            next_state_counts[prev_chain][next_state] += 1

            # Track total counts for this chain
            total_counts[prev_chain] = total_counts.get(prev_chain, 0) + 1

        # Convert counts to probabilities
        chain_probs = {}
        for prev_chain, state_counts in next_state_counts.items():
            chain_probs[prev_chain] = {
                state: count / total_counts[prev_chain]
                for state, count in state_counts.items()
            }

        full_cond_probs[chain_len] = chain_probs

    return full_cond_probs

# end of conditional_probabilities.py


# === FILE: src/data_fetcher.py ===

# ./src/data_fetcher.py 

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


def normalize_columns(data, symbol):
    """
    Renames columns with symbol-specific suffixes to generic OHLCV columns.

    E.g., from 'close_ng=f' to 'close'.
    """
    symbol_prefix = symbol.lower()

    rename_map = {
        f'open_{symbol_prefix}': 'open',
        f'high_{symbol_prefix}': 'high',
        f'low_{symbol_prefix}': 'low',
        f'close_{symbol_prefix}': 'close',
        f'adj_close_{symbol_prefix}': 'adj_close',
        f'volume_{symbol_prefix}': 'volume'
    }

    # Perform rename
    data.rename(columns=rename_map, inplace=True)

    # Optional sanity check
    required_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"[Error] Missing expected columns after renaming: {missing}")

    return data


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

    # Normalize column to 'date'
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

    data = normalize_columns(data, symbol)

    # Display first few rows for verification
    print(f"[Info] Data fetched successfully for {symbol}:\n")

    # [Info]: Show dataset description
    print("\n[Info] Dataset Metadata:")
    print(f"- Timeframe: {interval}")
    print(f"- Date Range: {start_date} to {end_date}")
    print(f"- Rows: {len(data)}")
    print(f"- Columns ({len(data.columns)}): {list(data.columns)}")
    print(f"- First Date: {data['date'].iloc[0] if not data.empty else 'N/A'}")
    print(f"- Last Date: {data['date'].iloc[-1] if not data.empty else 'N/A'}")

    # Preview first and last 3 rows (optional for deeper debugging)
    print("\n[Preview] First 3 rows:\n", data.head())
    print("\n[Preview] Last 3 rows:\n", data.tail())

    return data


# End of data_fetcher.py


# === FILE: src/descriptive_analysis.py ===

# ./src/descriptive_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def compute_central_tendency_dispersion(df, column):
    """
    Compute central tendency and dispersion metrics for a given column
    """
    metrics = {}
    data = df[column].dropna()

    metrics['mean'] = data.mean()
    metrics['median'] = data.median()
    metrics['mode'] = data.mode().iloc[0] if not data.mode().empty else None
    metrics['std_dev'] = data.std()
    metrics['variance'] = data.var()
    metrics['skewness'] = data.skew()
    metrics['kurtosis'] = data.kurtosis()

    print(f"\n[Info] Descriptive Stats for {column}:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics

def perform_normality_tests(df, column):
    """
    Perform normality tests on a given column
    """
    data = df[column].dropna()

    print(f"\n[Info] Normality Tests for {column}:")
    
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk Test: statistic={shapiro_stat}, p-value={shapiro_p}")

    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f"Kolmogorov-Smirnov Test: statistic={ks_stat}, p-value={ks_p}")

    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p
    }

def plot_distribution(df, column):
    """
    Plot histogram, density plot, and QQ-plot for a given column
    """
    data = df[column].dropna()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"Histogram and Density Plot for {column}")

    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"QQ-Plot for {column}")

    plt.tight_layout()
    plt.show()

def descriptive_statistics_pipeline(df, columns_to_analyze, show_plots=True):
    """
    Run descriptive analysis pipeline on a list of columns
    """
    print("[Info] Starting Descriptive Statistical Analysis...")
    results = {}

    for column in columns_to_analyze:
        print(f"\n--- Analyzing column: {column} ---")
        summary_stats = compute_central_tendency_dispersion(df, column)
        normality_tests = perform_normality_tests(df, column)
        if show_plots:
            plot_distribution(df, column)

        results[column] = {
            'summary': summary_stats,
            'normality': normality_tests
        }

    print("[Info] Descriptive Statistical Analysis Completed!")
    return results



# === FILE: src/distribution_analysis.py ===

# ./src/distribution_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner outputs
import warnings
warnings.filterwarnings('ignore')

def fit_distribution(data, dist_name):
    """
    Fit a distribution to the data using MLE and return the distribution object and parameters.
    """
    if dist_name == 'normal':
        dist = stats.norm
    elif dist_name == 'lognormal':
        dist = stats.lognorm
    elif dist_name == 'student_t':
        dist = stats.t
    else:
        raise ValueError(f"Distribution '{dist_name}' not supported.")
    
    params = dist.fit(data)
    return dist, params

def plot_distribution_fit(data, dist, params, dist_name, column):
    """
    Plot histogram with fitted PDF and Empirical CDF vs Fitted CDF.
    """
    x = np.linspace(min(data), max(data), 1000)
    
    # PDF
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=30, kde=False, stat='density', label='Empirical', color='skyblue')
    plt.plot(x, dist.pdf(x, *params), 'r-', label=f'{dist_name} PDF')
    plt.title(f"{column}: Histogram with {dist_name} PDF")
    plt.legend()

    # CDF
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, ecdf, marker='.', linestyle='none', label='Empirical CDF')
    plt.plot(x, dist.cdf(x, *params), 'r-', label=f'{dist_name} CDF')
    plt.title(f"{column}: Empirical vs {dist_name} CDF")
    plt.legend()

    plt.tight_layout()
    plt.show()

def goodness_of_fit(data, dist, params):
    """
    Perform Goodness-of-Fit tests: Chi-Square and Anderson-Darling.
    """
    # Chi-Square Test
    observed_freq, bins = np.histogram(data, bins='auto')
    expected_freq = len(data) * np.diff(dist.cdf(bins, *params))
    chi_square_stat = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()

    # Anderson-Darling Test
    ad_test = stats.anderson(data, dist='norm')  # Adjust as necessary for non-normal

    print(f"\n[GoF] Chi-Square Statistic: {chi_square_stat}")
    print(f"[GoF] Anderson-Darling Statistic: {ad_test.statistic}")

    return {
        'chi_square': chi_square_stat,
        'anderson_darling': ad_test.statistic
    }

def probability_of_move(dist, params, threshold, tail='both'):
    """
    Calculate the probability of a move greater than a threshold.
    """
    if tail == 'both':
        prob = dist.sf(threshold, *params) + dist.cdf(-threshold, *params)
    elif tail == 'upper':
        prob = dist.sf(threshold, *params)
    elif tail == 'lower':
        prob = dist.cdf(-threshold, *params)
    else:
        raise ValueError("tail must be 'both', 'upper', or 'lower'.")

    print(f"\n[Probability] Probability of move beyond ±{threshold}: {prob:.4f}")
    return prob

# default test thresholds = 2  or 2% move
def distribution_analysis_pipeline(df, column, thresholds=[2.00], show_plots=True):
    """
    Run distribution fitting, GoF tests, and probability estimation on a given column.
    """
    print(f"\n[Info] Starting Distribution Analysis for {column}...")
    
    data = df[column].dropna()
    results = {}

    # Fit and analyze each distribution
    for dist_name in ['normal', 'lognormal', 'student_t']:
        print(f"\n--- Fitting {dist_name} distribution ---")

        # Fit
        dist, params = fit_distribution(data, dist_name)

        # Plot PDF and CDF
        if show_plots:
            plot_distribution_fit(data, dist, params, dist_name, column)

        # GoF Tests
        gof_results = goodness_of_fit(data, dist, params)

        # Probability estimates
        prob_results = {}
        for threshold in thresholds:
            prob = probability_of_move(dist, params, threshold)
            prob_results[f'±{threshold}'] = prob

        # Store results
        results[dist_name] = {
            'params': params,
            'gof': gof_results,
            'probabilities': prob_results
        }

    print(f"\n[Info] Distribution Analysis Completed for {column}!")
    return results



# === FILE: src/filters.py ===

# ./src/filters.py
import pandas as pd
import numpy as np

def adaptive_outlier_filter(
    df,
    column='daily_return_%',
    method='combined',
    z_thresh=3.0,
    lower_quantile=0.01,
    upper_quantile=0.99,
    audit_filename='excluded_outliers.csv'
):
    """
    Adaptive outlier filter supporting z-score, quantile, or both.
    
    Parameters:
    - df: DataFrame containing data
    - column: column to filter (default = 'daily_return_%')
    - method: 'zscore', 'quantile', or 'combined' (default)
    - z_thresh: threshold for z-score method (default = 3.0)
    - lower_quantile: lower threshold for quantile method (default = 0.01)
    - upper_quantile: upper threshold for quantile method (default = 0.99)
    - audit_filename: filename to save excluded outliers

    Returns:
    - df_filtered: cleaned DataFrame
    """
    df = df.copy()
    
    # Start with no outlier flags
    df['is_outlier'] = False
    
    # Z-SCORE FILTERING
    if method in ['zscore', 'combined']:
        mean_val = df[column].mean()
        std_val = df[column].std()

        df['z_score'] = (df[column] - mean_val) / std_val

        # Mark z-score outliers
        df.loc[df['z_score'].abs() > z_thresh, 'is_outlier'] = True
        
        print(f"[Filter] Z-Score outlier filter applied: Threshold = {z_thresh}")

    # QUANTILE FILTERING
    if method in ['quantile', 'combined']:
        lower_bound = df[column].quantile(lower_quantile)
        upper_bound = df[column].quantile(upper_quantile)

        # Mark quantile outliers
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), 'is_outlier'] = True
        
        print(f"[Filter] Quantile outlier filter applied: "
              f"Lower {lower_quantile*100:.1f}%, Upper {upper_quantile*100:.1f}%")

    # SAVE EXCLUDED OUTLIERS
    outliers = df[df['is_outlier']].copy()
    if not outliers.empty:
        outliers.to_csv(audit_filename, index=False)
        print(f"[Audit] {len(outliers)} outlier rows saved to '{audit_filename}'")

    # CLEAN DATAFRAME
    df_filtered = df[~df['is_outlier']].copy()

    # Drop temporary columns
    df_filtered.drop(columns=['is_outlier'], inplace=True)
    if 'z_score' in df_filtered.columns:
        df_filtered.drop(columns=['z_score'], inplace=True)

    print(f"[Filter] Cleaned dataset: {len(df_filtered)} rows (removed {len(outliers)} outliers)")

    return df_filtered

# end of filters.py


# === FILE: src/label_candlesticks.py ===

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


# === FILE: src/monte_carlo.py ===

# ./src/monte_carlo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# --------------------------------------------------------
# 1. Basic Monte Carlo Simulation (Log Returns Model)
# --------------------------------------------------------
def monte_carlo_simulation(start_price, mu, sigma, n_steps, n_simulations, random_seed=42):
    """
    Perform a basic Monte Carlo simulation for future price paths.
    Args:
        start_price: initial price
        mu: expected return (mean)
        sigma: volatility (std deviation)
        n_steps: number of time steps to simulate
        n_simulations: number of simulation paths
    Returns:
        simulated_price_paths: (n_steps + 1, n_simulations) array
    """
    np.random.seed(random_seed)
    dt = 1  # 1 day time increment (daily returns)

    # Generate random returns
    random_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=(n_steps, n_simulations))

    # Convert returns to price paths
    price_paths = np.zeros((n_steps + 1, n_simulations))
    price_paths[0] = start_price

    for t in range(1, n_steps + 1):
        price_paths[t] = price_paths[t - 1] * np.exp(random_returns[t - 1])

    print(f"Function monte_carlo_simulation executed {n_simulations} simulations price paths with {n_steps} steps each.")
    print(f"Start Price: {start_price}, Mu: {mu:.5f}, Sigma: {sigma:.5f}")
    print(f"Simulated Price Paths: {price_paths.shape}")

    return price_paths

# --------------------------------------------------------
# 2. Plot Simulated Paths
# --------------------------------------------------------
def plot_simulations(price_paths):
    """
    Plot simulated price paths.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(price_paths, lw=0.8, alpha=0.6)
    plt.title("Monte Carlo Simulated Price Paths")
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Price")
    plt.show()

# --------------------------------------------------------
# 3. Calculate VaR and Expected Shortfall (CVaR)
# --------------------------------------------------------
def calculate_var_cvar(price_paths, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (CVaR) from simulated returns.
    """
    # Calculate ending prices and returns from the simulations
    ending_prices = price_paths[-1]
    returns = (ending_prices - price_paths[0, 0]) / price_paths[0, 0]

    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()

    print(f"[Risk: funciton monte_carlo_simulation] VaR ({confidence_level * 100:.0f}%): {var * 100:.2f}%")
    print(f"[Risk: function monte_carlo_simulation] CVaR ({confidence_level * 100:.0f}%): {cvar * 100:.2f}%")

    return var, cvar

# end of monte_carlo.py


# === FILE: src/monte_carlo_v2.py ===

# ./src/monte_carlo_simulation_v2.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# --------------------------------------------------------
# 1. Adaptive Monte Carlo Simulation (Regime & Volatility-Aware)
# --------------------------------------------------------
def monte_carlo_simulation_v2(start_price, regimes_df, garch_results, hmm_model, n_steps=30, n_simulations=500, random_seed=42):
    """
    Monte Carlo Simulation using regime-specific mu/sigma and GARCH volatility forecasts.
    
    Args:
        start_price: initial price
        regimes_df: dataframe with regime states assigned (output from HMM)
        garch_results: fitted GARCH model for volatility forecasts
        hmm_model: fitted HMM model for regime probabilities
        n_steps: simulation length (days)
        n_simulations: number of simulation paths
    Returns:
        simulated_price_paths: array of simulated price paths
    """

    np.random.seed(random_seed)

    # STEP 1: Calculate regime-specific mu/sigma
    # regime_states = regimes_df['hmm_state'].unique()
    regime_states = np.arange(hmm_model.n_components)  # <<< FIXED
    regime_mu_sigma = {}

    # Fallback values from full dataset if any regime state is missing or invalid
    emp_mu = regimes_df['daily_return_%'].mean() / 100
    emp_sigma = regimes_df['daily_return_%'].std() / 100

    for state in regime_states:
        subset = regimes_df[regimes_df['hmm_state'] == state]
        mu = subset['daily_return_%'].mean() / 100  # to decimal
        sigma = subset['daily_return_%'].std() / 100

        # Fallback to empirical values if missing
        if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
            print(f"[Warning] Regime {state} has invalid mu/sigma. Using fallback (empirical) values.")
            mu, sigma = emp_mu, emp_sigma

        regime_mu_sigma[state] = (mu, sigma)
        print(f"[Regime {state}] mu: {mu:.5f}, sigma: {sigma:.5f}")

    # STEP 2: Initialize Monte Carlo simulation paths
    price_paths = np.zeros((n_steps + 1, n_simulations))
    price_paths[0] = start_price

    # STEP 3: Use last known state as starting point
    last_state = regimes_df['hmm_state'].iloc[-1]

    # STEP 4: Use transition matrix for regime transitions
    trans_mat = hmm_model.transmat_

    # STEP 5: Simulate price paths
    for sim in range(n_simulations):
        current_price = start_price
        state = last_state

        for t in range(1, n_steps + 1):
            # Transition to next regime (using transition probabilities)
            state = np.random.choice(regime_states, p=trans_mat[state])

            # Regime-specific mu, sigma
            mu, sigma = regime_mu_sigma[state]

            # Optional: GARCH forecast overrides sigma (uncomment if you want dynamic volatility)
            # vol_forecast = garch_results.forecast(horizon=t).variance.iloc[-1, 0] ** 0.5
            # sigma = vol_forecast

            # Generate a random return
            random_return = np.random.normal(mu, sigma)

            # Price evolution
            current_price *= np.exp(random_return)
            price_paths[t, sim] = current_price
    print(f"Function monte_carlo_simulation_v2 executed {n_simulations} simulations price paths with {n_steps} steps each.")
    print(f"Start Price: {start_price}")
    print(f"Simulated Price Paths: {price_paths.shape}")

    return price_paths

# --------------------------------------------------------
# 2. Plot Simulated Paths
# --------------------------------------------------------
def plot_simulations(price_paths, title="Monte Carlo Simulated Price Paths (Regime + Volatility Aware)"):
    """
    Plot simulated price paths.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(price_paths, lw=0.8, alpha=0.6)
    plt.title(title)
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Price")
    plt.show()

# --------------------------------------------------------
# 3. Calculate VaR and Expected Shortfall (CVaR)
# --------------------------------------------------------
def calculate_var_cvar(price_paths, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (CVaR) from simulated returns.
    """
    ending_prices = price_paths[-1]
    returns = (ending_prices - price_paths[0, 0]) / price_paths[0, 0]

    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()

    print(f"[Risk: function monte_carlo_simulation_v2] VaR ({confidence_level * 100:.0f}%): {var * 100:.2f}%")
    print(f"[Risk: function monte_carlo_simulation_V2] CVaR ({confidence_level * 100:.0f}%): {cvar * 100:.2f}%")

    return var, cvar


# end of monte_carlo_simulation_v2.py


# === FILE: src/report_generator.py ===

# ./src/report_generator.py

import os
from datetime import datetime

# Ensure reports folder exists
REPORTS_FOLDER = 'reports'
os.makedirs(REPORTS_FOLDER, exist_ok=True)


def continuation_probability_lookup(data, conditional_probs, column_state='state'):
    """
    Looks up the continuation probability for the latest chain of states in the dataset.
    
    Args:
        data (pd.DataFrame): The data containing the 'state' column.
        conditional_probs (dict): The output from conditional_probabilities_pipeline(),
                                  should include 'cond_probs' key.
    
    Returns:
        result (dict): {
            'pattern': string representation of the chain,
            'probability': continuation probability as a float (0-1),
            'message': friendly text summary (optional)
        }
    """
    
    chain_probs = conditional_probs.get('cond_probs', {})

    if not chain_probs:
        return {
            'pattern': None,
            'probability': None,
            'message': "No conditional probabilities available."
        }

    max_chain_len = max(chain_probs.keys(), default=0)

    if max_chain_len == 0:
        return {
            'pattern': None,
            'probability': None,
            'message': "No valid chains found in conditional probabilities."
        }

    # Extract the latest chain from the data (up to the max_chain_len)
    state_chain = tuple(data[column_state].iloc[-max_chain_len:])

    probability = None
    matched_chain_str = None

    # Search for the longest possible matching chain
    for chain_len in range(max_chain_len, 0, -1):
        lookup_chain = state_chain[-chain_len:]
        lookup_probs = chain_probs.get(chain_len, {})

        if lookup_chain in lookup_probs:
            probability = lookup_probs[lookup_chain]
            matched_chain_str = " > ".join(lookup_chain)
            break

    if probability is not None:
        message = (
            f"Pattern: {matched_chain_str}\n"
            f"Continuation Probability: {probability * 100:.2f}%"
        )
    else:
        message = "No continuation probability available for current pattern."

    return {
        'pattern': matched_chain_str,
        'probability': probability,
        'message': message
    }


def conditional_probability_lookup_full(conditional_probs, data, report_lines=None, column_state='state'):
    """
    Enhanced lookup that reports continuation probability and full next state probabilities.
    
    Args:
        conditional_probs (dict): Output from conditional_probabilities_pipeline()
        data (DataFrame): The full dataset containing 'state' column
        report_lines (list): Optionally pass a report_lines list to append output (for report)
    
    Returns:
        continuation_prob (float): Continuation probability found (or None)
    """
    # Use existing report_lines or create a new one
    local_lines = report_lines if report_lines is not None else []

    chain_probs = conditional_probs.get('cond_probs', {})
    tpm_df = conditional_probs.get('tpm', None)
    
    max_chain_len = max(chain_probs.keys(), default=0)

    # Get the last N states from the data (up to max_chain_len)
    state_chain = tuple(data[column_state].iloc[-max_chain_len:])

    continuation_prob = None  # Return value
    found = False  # Control flag for loop

    # Start from the longest chain and work backward until we find something
    for chain_len in range(max_chain_len, 0, -1):
        lookup_chain = state_chain[-chain_len:]
        lookup_probs = chain_probs.get(chain_len, {})

        if lookup_chain in lookup_probs:
            continuation_prob = lookup_probs.get(lookup_chain, 0)
            found = True

            local_lines.append("\n>>> Continuation Probability Lookup")
            local_lines.append(f"Pattern: {' > '.join(lookup_chain)}")
            local_lines.append(f"Continuation Probability (stay {lookup_chain[-1]}): {continuation_prob * 100:.2f}%")

            break  # Found the longest matching chain; no need to continue

    if not found:
        local_lines.append("No continuation probability available for current pattern.")
        return continuation_prob

    # === NEXT STATE PROBABILITIES ===
    last_state = lookup_chain[-1]
    if tpm_df is not None and last_state in tpm_df.index:
        tpm_row = tpm_df.loc[last_state]

        local_lines.append(f"\n>>> Probabilities for Next State (from state '{last_state}'):")
        for next_state, prob in tpm_row.items():
            label = f"- {next_state}"
            # Highlight reversal if it is different from continuation
            if next_state != last_state:
                label += " (reversal)"
            local_lines.append(f"{label}: {prob * 100:.2f}%")

    # === Optional: Return probability or report ===
    return continuation_prob


def full_conditional_probability_lookup_full(data, full_cond_probs, state_column='state'):
    """
    Look up continuation probability + full next state probabilities for the current state chain.
    
    Args:
        data (DataFrame): must have 'state' column with classified states.
        full_cond_probs (dict): output from full_conditional_probabilities()

    Returns:
        list: formatted report lines (strings)
    """
    report_lines = []
    
    # Get the maximum chain length we calculated
    max_chain_len = max(full_cond_probs.keys(), default=0)

    # Get the latest chain from data
    state_chain = tuple(data[state_column].iloc[-max_chain_len:])

    continuation_prob = None
    next_state_probs = None

    # Start from longest chain length and go backward
    for chain_len in range(max_chain_len, 0, -1):
        lookup_chain = state_chain[-chain_len:]
        chain_probs = full_cond_probs.get(chain_len, {})

        if lookup_chain in chain_probs:
            next_state_probs = chain_probs[lookup_chain]
            continuation_prob = next_state_probs.get(lookup_chain[-1], None)

            # chain_str = " > ".join(lookup_chain)
            chain_str = " > ".join(map(str, lookup_chain))

            report_lines.append(f"Pattern: {chain_str}")

            report_lines.append("Next State Probabilities:")
            for state, prob in sorted(next_state_probs.items(), key=lambda x: x[1], reverse=True):
                direction = "continuation" if state == lookup_chain[-1] else "reversal"
                report_lines.append(f"- {state} ({direction}): {prob * 100:.2f}%")

            break

    if next_state_probs is None:
        report_lines.append("No continuation probabilities found for current pattern.")

    return report_lines


def format_chain(chain):
    return " > ".join(map(str, chain))


def full_conditional_probability_lookup_full_verbose(data, full_cond_probs, state_column='state'):
    """
    Look up continuation probability + full next state probabilities for the current state chain.
    
    Args:
        data (DataFrame): must have 'state' column with classified states.
        full_cond_probs (dict): output from full_conditional_probabilities()

    Returns:
        tuple(continuation_prob, next_state_probs)
    """
    # Get the maximum chain length we calculated
    max_chain_len = max(full_cond_probs.keys(), default=0)

    # Get the latest chain from data
    state_chain = tuple(data[state_column].iloc[-max_chain_len:])

    continuation_prob = None
    next_state_probs = None

    # Start from longest chain length and go backward
    for chain_len in range(max_chain_len, 0, -1):
        lookup_chain = state_chain[-chain_len:]
        chain_probs = full_cond_probs.get(chain_len, {})

        if lookup_chain in chain_probs:
            next_state_probs = chain_probs[lookup_chain]
            # Get continuation probability specifically for last state
            continuation_prob = next_state_probs.get(lookup_chain[-1], None)

            # For verbose display (optional)
            print(f"\n[Verbose] Probabilities for Next State (Chain Lookup):")
            # chain_str = " > ".join(lookup_chain)
            chain_str = " > ".join(map(str, lookup_chain))
            print(f"Pattern: {chain_str}")

            for state, prob in sorted(next_state_probs.items(), key=lambda x: x[1], reverse=True):
                direction = "continuation" if state == lookup_chain[-1] else "reversal"
                print(f"- {state} ({direction}): {prob * 100:.2f}%")

            break

    return continuation_prob, next_state_probs



def format_trade_signal(trade_signal, symbol, timeframe):
    """
    Returns a formatted string of the trade signal for CLI or report use.
    """
    action = trade_signal.get('trade_action', 'HOLD')

    execution_details = f"""
• Entry Type        : {trade_signal.get('entry_type')}
• Entry Price       : {trade_signal.get('entry_price')}

• Stop Loss Price   : {trade_signal.get('stop_loss_price')}
• Take Profit Price : {trade_signal.get('take_profit_price')}

• Stop Loss Distance   : {trade_signal.get('stop_loss_distance')}
• Take Profit Distance : {trade_signal.get('take_profit_distance')}

• Recommended Position : {trade_signal.get('position_size')} lots
"""

    report = f"""
=======================================
           TRADE SIGNAL SUMMARY         
=======================================
• Symbol            : {symbol}
• Timeframe         : {timeframe}
• Current Regime    : {trade_signal.get('regime')}
• Strategy Bias     : {trade_signal.get('bias')}
• Action            : {action}
• Confidence Level  : {trade_signal.get('confidence')}

---------------------------------------
         EXECUTION PARAMETERS         
---------------------------------------
{execution_details}
=======================================
"""
    return report


# def format_latest_candles(data, n=5, state_column='state'):
#     rows = []
#     rows.append("Latest Candles Overview")
#     rows.append("─" * 55)
#     rows.append("Date        |  O     H     L     C  |  State  |")
#     rows.append("────────────┼───────────────────────┼─────────┤")
#     
#     for _, row in data.tail(n).iterrows():
#         date = row['date'].strftime("%Y-%m-%d")
#         o, h, l, c = [f"{row[col]:.2f}" for col in ['open', 'high', 'low', 'close']]
#         # state = row.get('candle_state', 'N/A')
#         state = row.get(state_column, 'N/A')
#         rows.append(f"{date}  | {o} {h} {l} {c} | {state:<6} |")
#
#     rows.append("─" * 55)
#     sma_fast = data['sma_fast'].iloc[-1]
#     sma_slow = data['sma_slow'].iloc[-1]
#     last_row = data.iloc[-1]
#     rows.append(f"SMA Fast: {sma_fast:.2f}     SMA Slow: {sma_slow:.2f}")
#     rows.append(f"Current State: {last_row.get('candle_state', 'N/A')}  | Candle Pattern: {last_row.get('candlestick', 'N/A')}")
#     rows.append("─" * 55)
#     
#     return "\n".join(rows)


def format_latest_candles(data, n=5, state_column='state'):
    rows = []
    rows.append(">>> Latest Candles Overview")
    rows.append("─" * 70)
    rows.append("Date        |   O     H     L     C   |   %   |  State  |")
    rows.append("────────────┼─────────────────────────┼───────┼─────────┤")
    
    for _, row in data.tail(n).iterrows():
        date = row['date'].strftime("%Y-%m-%d")
        o, h, l, c = [f"{row[col]:.2f}" for col in ['open', 'high', 'low', 'close']]
        pct = row.get('daily_return_%', 0.0)
        pct_str = f"{pct:+.2f}%"
        state = row.get(state_column, 'N/A')
        rows.append(f"{date}  | {o} {h} {l} {c} | {pct_str:>6} | {state:<6} |")

    rows.append("─" * 70)
    sma_fast = data['sma_fast'].iloc[-1]
    sma_slow = data['sma_slow'].iloc[-1]
    last_row = data.iloc[-1]
    rows.append(f"SMA Fast: {sma_fast:.2f}     SMA Slow: {sma_slow:.2f}")
    rows.append(f"Current State: {last_row.get('candle_state', 'N/A')}  | Candle Pattern: {last_row.get('candlestick', 'N/A')}")
    rows.append("─" * 70)

    return "\n".join(rows)


def generate_report(
    symbol,
    timeframe,
    start_date,
    end_date,
    data,
    descriptive_stats,
    distribution_results,
    conditional_probs,
    full_cond_probs,
    garch_results,
    volatility_forecast,  # <-- here it is!
    hmm_results,
    monte_carlo,
    trade_signal,
    state_column='state'):
    """
    Generates a comprehensive report and saves it to file.
    """
    report_lines = []

    # === HEADER ===
    report_lines.append("=" * 50)
    report_lines.append(f"{'RISK ANALYZER FINAL REPORT':^50}")
    report_lines.append("=" * 50)
    report_lines.append(f"Symbol        : {symbol}")
    report_lines.append(f"Timeframe     : {timeframe}")
    report_lines.append(f"Date Range    : {start_date} to {end_date}")
    report_lines.append(f"Data Points   : {len(data)}")
    report_lines.append("-" * 50)

    # === CURRENT MARKET SNAPSHOT ===
    report_lines.append("\n>>> Current Market Snapshot")

    # === CONTINUATION PROBABILITY LOOKUP ===

    # Latest close price
    latest_close = data['close'].iloc[-1] if 'close' in data.columns else 'N/A'

    # Latest state (bullish, bearish, neutral) from conditional probabilities
    latest_state = data[state_column].iloc[-1] if 'state' in data.columns else 'N/A'

    report_lines.append(f"- Latest Price Close      : {latest_close:,.2f} USD")
    report_lines.append(f"- Latest Return State     : {latest_state}")
    report_lines.append("\n>>> Continuation Probability Lookup")

    # === FULL CONDITIONAL PROBABILITIES ===

    # You pass the current dataset and your full_cond_probs dictionary here!
    full_lookup_result = full_conditional_probability_lookup_full(
        data,
        full_cond_probs,
        state_column=state_column)

    print("[Debug] Available columns:", data.columns.tolist())
    print("[Debug] Requested state column:", state_column)

    report_lines.extend(full_lookup_result)

    # === DESCRIPTIVE STATISTICS ===
    report_lines.append("\n>>> Descriptive Statistics (daily_return_%)")
    desc = descriptive_stats.get('daily_return_%', {}).get('summary', {})
    report_lines.append(f"Mean          : {desc.get('mean', 'N/A'):.4f} %")
    report_lines.append(f"Std Dev       : {desc.get('std_dev', 'N/A'):.4f} %")
    report_lines.append(f"Skewness      : {desc.get('skewness', 'N/A'):.4f}")
    report_lines.append(f"Kurtosis      : {desc.get('kurtosis', 'N/A'):.4f}")

    # === DISTRIBUTION ANALYSIS ===
    report_lines.append("\n>>> Distribution Analysis (Normal Distribution)")
    dist = distribution_results.get('normal', {}).get('probabilities', {})
    for threshold, prob in dist.items():
        report_lines.append(f"P(|move| > {threshold}) : {prob*100:.2f} %")

    # === CONDITIONAL PROBABILITIES ===
    report_lines.append("\n>>> Conditional Probabilities")
    runs = conditional_probs.get('runs_info', {})
    report_lines.append(f"Total Runs    : {runs.get('total_runs', 'N/A')}")
    # report_lines.append(f"Z-Score       : {runs.get('z_score', 'N/A'):.2f}")
    z_score = runs.get("z_score", None)
    if isinstance(z_score, (float, int)):
        report_lines.append(f"Z-Score       : {z_score:.2f}")
    else:
        report_lines.append("Z-Score       : N/A")

    # Conditional chains
    chains = conditional_probs.get('cond_probs', {}).get(5, {})
    chain_items = list(chains.items())
    n = 3
    report_lines.append(f"\n>>> Top/Tail {n} Conditional Chains (Chain Length: 5)")
    for chain, prob in chain_items[:n]:
        # chain_str = " > ".join(chain)
        chain_str = " > ".join(map(str, chain))
        report_lines.append(f"{chain_str} => Continuation: {prob*100:.2f}%")

    # ... elipsis 
    if len(chains) > n:
        report_lines.append("...")

    for chain, prob in chain_items[-n:]:
        # chain_str = " > ".join(chain)
        chain_str = " > ".join(map(str, chain))
        report_lines.append(f"{chain_str} => Continuation: {prob*100:.2f}%")

    # === VOLATILITY FORECAST ===
    report_lines.append("\n>>> Volatility Forecast (Next 5 steps)")

    try:
        for idx, row in volatility_forecast.iterrows():
            step = idx + 1
            # Assuming the forecast_volatility column is already decimal, convert to percentage
            vol_pct = row['forecast_volatility'] * 100
            report_lines.append(f"Step {step}: {vol_pct:.2f}%")
    except Exception as e:
        report_lines.append(f"Volatility Forecast Error: {e}")

    # === HMM STATES ===
    report_lines.append("\n>>> HMM Regimes (Means and Covariances)")
    try:
        for idx, mean in enumerate(hmm_results.means_.flatten()):
            cov = hmm_results.covars_.flatten()[idx]
            report_lines.append(f"State {idx}: Mean={mean:.6f}, Cov={cov:.6f}")
    except Exception as e:
        report_lines.append(f"HMM Results Error: {e}")

    # === MONTE CARLO SIMULATION RESULTS ===
    report_lines.append("\n>>> Monte Carlo Simulation Results (VaR and CVaR)")

    try:
        v1 = monte_carlo.get('v1', {})
        v2 = monte_carlo.get('v2', {})

        report_lines.append("\n-- Simple Monte Carlo (v1):")
        report_lines.append(f"VaR (95%): {v1.get('var', 'N/A')*100:.2f} %")
        report_lines.append(f"CVaR (95%): {v1.get('cvar', 'N/A')*100:.2f} %")

        report_lines.append("\n-- Regime & Volatility Aware Monte Carlo (v2):")
        report_lines.append(f"VaR (95%): {v2.get('var', 'N/A')*100:.2f} %")
        report_lines.append(f"CVaR (95%): {v2.get('cvar', 'N/A')*100:.2f} %")

    except Exception as e:
        report_lines.append(f"Monte Carlo Section Error: {e}")

    # === SIMPLE TRADE SIGNAL ===
    report_lines.append("\n>>> Trade Signal Recommendation")
    report_lines.append(f"Regime        : {trade_signal.get('regime', 'N/A')}")
    report_lines.append(f"Bias          : {trade_signal.get('bias', 'N/A')}")
    report_lines.append(f"Action        : {trade_signal.get('trade_action', 'N/A')}")
    report_lines.append(f"Confidence    : {trade_signal.get('confidence', 'N/A')}")
    report_lines.append(f"Position Size : {trade_signal.get('position_size', 'N/A')} lots")

    if trade_signal.get('trade_action') != "HOLD":
        report_lines.append(f"Entry Price   : {trade_signal.get('entry_price', 'N/A')}")
        report_lines.append(f"Stop Loss     : {trade_signal.get('stop_loss_price', 'N/A')}")
        report_lines.append(f"Take Profit   : {trade_signal.get('take_profit_price', 'N/A')}")

    # === DETAILED TRADE PARAMETERS ===
    trade_signal_report = format_trade_signal(trade_signal, symbol, timeframe)
    report_lines.append("\n>>> Detailed Trade Signal")
    report_lines.append(trade_signal_report)

    # === LATEST CANDLES ===
    report_lines.append(format_latest_candles(data, n=5, state_column=state_column))

    # === FOOTER ===
    report_lines.append("-" * 50)
    report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50)

    # JOIN all lines
    # report_content = "\n".join(report_lines)
    report_content = "\n".join(str(line) for line in report_lines)

    # === PRINT TO CONSOLE ===
    print("\n" + report_content + "\n")

    # === SAVE TO FILE ===
    file_name = f"{symbol.replace('=','_')}_{timeframe}_{end_date}_report.txt"
    file_path = os.path.join(REPORTS_FOLDER, file_name)

    with open(file_path, 'w') as file:
        file.write(report_content)

    print(f"[Info] Report saved to {file_path}")

# end of report_generator.py


# === FILE: src/statistical_analysis.py ===

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



# === FILE: src/trading_rules.py ===

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


def determine_trade_signal(regime,
                           current_price,
                           mean_price,
                           contract_size,
                           continuation_prob,
                           reversal_prob,
                           garch_vol,
                           account_size=100000,
                           risk_per_trade=0.01):
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
    per_contract_risk = stop_loss_distance * contract_size
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
        position_size_factor = risk_per_trade * account_size * contract_size / (2 * volatility_dollar_move)
    else:
        position_size_factor = 0

    if per_contract_risk != 0:
        position_size = (risk_per_trade * account_size) / per_contract_risk
    else:
        position_size = 0

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
        # "position_size": round(position_size_factor, 2),
        "position_size": round(position_size, 2),
        "confidence": confidence
    }

    return trade_signal

# end of trading_rules.py


# === FILE: src/volatility_analysis.py ===

# ./src/volatility_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

sns.set(style='whitegrid')

# --------------------------------------------------------
# 1. Fit GARCH(1,1) Model
# --------------------------------------------------------
def fit_garch(df, return_column='daily_return_%', p=1, q=1):
    returns = df[return_column].dropna()/100

    # Rescale returns for intraday (optional but useful)
    # returns_rescaled = returns #* 100

    am = arch_model(returns, p=p, q=q, mean='Zero', vol='GARCH', dist='normal')
    model = am.fit(disp='off')

    if model.convergence_flag != 0:
        print(f"WARNING: GARCH model did not fully converge (code {model.convergence_flag})")

    cond_volatility = model.conditional_volatility #/100

    return model


# --------------------------------------------------------
# 2. Plot Volatility Clusters (Conditional Volatility)
# --------------------------------------------------------
def plot_volatility(df, garch_results, return_column='daily_return_%'):
    # Get forecast volatility
    forecast_vol = garch_results.conditional_volatility

    # Check for time axis column
    time_axis = None
    for col in ['date', 'datetime', 'timestamp']:
        if col in df.columns:
            time_axis = df[col]
            break

    if time_axis is None:
        # Fallback to index
        time_axis = df.index

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, df[return_column], label='Returns')
    plt.title('Returns')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis[-len(forecast_vol):], forecast_vol, color='orange', label='GARCH Volatility')
    plt.title('GARCH Conditional Volatility')
    plt.legend()

    plt.tight_layout()
    plt.show()



# --------------------------------------------------------
# 3. Forecast Volatility
# --------------------------------------------------------
def forecast_volatility(garch_results, steps=5):
    forecasts = garch_results.forecast(horizon=steps)

    vol_forecasts = forecasts.variance.values[-1, :]

    # Convert variance to std dev and rescale from % to decimal
    forecast_vols = np.sqrt(vol_forecasts) #/ 100

    df_forecast = pd.DataFrame({
        'forecast_mean': [0.0]*steps,
        'forecast_volatility': forecast_vols
    })

    print("\n[Info] Volatility Forecast (Decimal % Move):\n", df_forecast.head())
    return df_forecast




# end of volatility_analysis.py


# === FILE: src/volatility_regimes.py ===

# ./src/volatility_regimes.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM

sns.set(style='whitegrid')

# --------------------------------------------------------
# 1. Fit HMM for Volatility Regimes
# --------------------------------------------------------
def fit_hmm(df, return_column='daily_return_%', n_states=2):
    """
    Fit a Gaussian Hidden Markov Model to the returns data.
    Args:
        df: DataFrame with return column.
        return_column: name of the return column (percentage returns).
        n_states: number of regimes (states) to classify.
    Returns:
        model: fitted HMM model.
        hidden_states: regime classifications per observation.
    """
    # Convert returns to numeric and drop NaNs
    returns = df[return_column].dropna()

    # Convert % returns to decimal returns if needed
    returns = returns / 100.0 if returns.abs().max() > 1 else returns

    # Reshape returns to 2D array (required by hmmlearn)
    X = returns.values.reshape(-1, 1)

    print(f"\n[Info] Fitting HMM with {n_states} states on {return_column}...")

    # Fit the Gaussian HMM
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, random_state=42)
    model.fit(X)

    # Predict the hidden states (regimes)
    hidden_states = model.predict(X)

    # Add hidden state back to DataFrame
    df = df.iloc[-len(hidden_states):].copy()
    df['hmm_state'] = hidden_states

    print(f"[Info] HMM transition matrix:\n{model.transmat_}")
    print(f"[Info] HMM means (per state): {model.means_.flatten()}")
    print(f"[Info] HMM covariances (per state): {model.covars_.flatten()}")

    # --------------------------------------------------
    # Classify regimes as Low, Moderate, High Volatility
    # --------------------------------------------------
    regime_states = range(model.n_components)

    # Step 1: Get volatility distribution thresholds
    vol_col = 'daily_range_%' if 'daily_range_%' in df.columns else return_column
    low_vol = df[vol_col].quantile(0.33)
    high_vol = df[vol_col].quantile(0.66)

    # Step 2: Calculate volatility per HMM state
    state_volatilities = {
        state: df[df['hmm_state'] == state][vol_col].std()
        for state in regime_states
    }

    # Step 3: Map volatility level to labels
    state_labels = {}
    for state, vol in state_volatilities.items():
        if vol < low_vol:
            label = "Low Volatility"
        elif vol > high_vol:
            label = "High Volatility"
        else:
            label = "Moderate Volatility"
        state_labels[state] = label
        print(f"[Regime {state}] volatility = {vol:.4f} → {label}")

    # Step 4: Add labeled column to DataFrame
    df['volatility_regime'] = df['hmm_state'].map(state_labels)

    return model, df

# --------------------------------------------------------
# 2. Plot Regimes Over Time
# --------------------------------------------------------
def plot_hmm_states(df, return_column='daily_return_%', date_column='date'):
    """
    Plot returns color-coded by HMM regimes.
    """
    states = df['hmm_state'].unique()
    colors = sns.color_palette('deep', len(states))

    plt.figure(figsize=(14, 6))
    for state, color in zip(states, colors):
        subset = df[df['hmm_state'] == state]
        plt.scatter(subset[date_column], subset[return_column], label=f"State {state}", color=color, s=10)

    plt.title('Returns Classified by HMM States')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.show()

# --------------------------------------------------------
# 3. Plot State Probabilities Over Time
# --------------------------------------------------------
def plot_state_probabilities(model, df, date_column='date'):
    """
    Plot the probability of each state over time.
    """
    X = df['daily_return_%'].values.reshape(-1, 1)
    prob_states = model.predict_proba(X)

    plt.figure(figsize=(14, 6))
    for i in range(prob_states.shape[1]):
        plt.plot(df[date_column], prob_states[:, i], label=f"State {i}")

    plt.title("Probability of Regimes Over Time")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

# --------------------------------------------------------
# 4. Pipeline Function
# --------------------------------------------------------
def hmm_volatility_pipeline(df, return_column='daily_return_%', n_states=2, show_plots=True):
    """
    Full pipeline for fitting HMM, plotting regimes and probabilities.
    """
    print("\n[Info] Starting Hidden Markov Model (HMM) Regime Detection...")

    # Step 1: Fit HMM model
    model, df_with_states = fit_hmm(df, return_column=return_column, n_states=n_states)

    # Step 2: Plot states (scatter by regime)
    if show_plots:
        plot_hmm_states(df_with_states, return_column=return_column)

    # Step 3: Plot regime probabilities
    if show_plots:
        plot_state_probabilities(model, df_with_states)

    print("\n[Info] HMM Regime Detection Completed.")
    return model, df_with_states


# end of volatility_regimes.py
