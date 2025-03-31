# ./src/report_generator.py

import os
from datetime import datetime

# Ensure reports folder exists
REPORTS_FOLDER = 'reports'
os.makedirs(REPORTS_FOLDER, exist_ok=True)


def continuation_probability_lookup(data, conditional_probs):
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
    state_chain = tuple(data['state'].iloc[-max_chain_len:])

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


def conditional_probability_lookup_full(conditional_probs, data, report_lines=None):
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
    state_chain = tuple(data['state'].iloc[-max_chain_len:])

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


# def full_conditional_probability_lookup_full(data, full_cond_probs):
#     """
#     Look up the full probability distribution for the current state chain.
#     
#     Args:
#         data (DataFrame): must have 'state' column with classified states.
#         full_cond_probs (dict): output from full_conditional_probabilities()
#
#     Returns:
#         tuple(chain_str, probabilities_dict)
#     """
#     report_lines = []
#
#     # Get the maximum chain length we calculated
#     max_chain_len = max(full_cond_probs.keys(), default=0)
#
#     # Get the latest chain from data
#     state_chain = tuple(data['state'].iloc[-max_chain_len:])
#
#     # Placeholder for probabilities
#     continuation_prob = None
#     next_state_probs = None
#     probability_distribution = None
#
#     # Start from longest chain length and go backward
#     for chain_len in range(max_chain_len, 0, -1):
#         lookup_chain = state_chain[-chain_len:]
#         chain_probs = full_cond_probs.get(chain_len, {})
#
#         if lookup_chain in chain_probs:
#             probability_distribution = chain_probs[lookup_chain]
#             chain_str = " > ".join(lookup_chain)
#             report_lines.append(f"Pattern: {chain_str}")
#             report_lines.append(f"Next State Probabilities:")
#
#             # Sort by highest probability first
#             for state, prob in sorted(probability_distribution.items(), key=lambda x: x[1], reverse=True):
#                 if state == lookup_chain[-1]:
#                     label = f"{state} (continuation)"
#                 else:
#                     label = f"{state} (reversal)"
#
#                 report_lines.append(f"- {label}: {prob * 100:.2f}%")
#             break
#
#     if probability_distribution is None:
#         report_lines.append("No continuation probabilities found for current pattern.")
#
#     return report_lines, probability_distribution


def full_conditional_probability_lookup_full(data, full_cond_probs):
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
    state_chain = tuple(data['state'].iloc[-max_chain_len:])

    continuation_prob = None
    next_state_probs = None

    # Start from longest chain length and go backward
    for chain_len in range(max_chain_len, 0, -1):
        lookup_chain = state_chain[-chain_len:]
        chain_probs = full_cond_probs.get(chain_len, {})

        if lookup_chain in chain_probs:
            next_state_probs = chain_probs[lookup_chain]
            continuation_prob = next_state_probs.get(lookup_chain[-1], None)

            chain_str = " > ".join(lookup_chain)
            report_lines.append(f"Pattern: {chain_str}")

            report_lines.append("Next State Probabilities:")
            for state, prob in sorted(next_state_probs.items(), key=lambda x: x[1], reverse=True):
                direction = "continuation" if state == lookup_chain[-1] else "reversal"
                report_lines.append(f"- {state} ({direction}): {prob * 100:.2f}%")

            break

    if next_state_probs is None:
        report_lines.append("No continuation probabilities found for current pattern.")

    return report_lines


def full_conditional_probability_lookup_full_verbose(data, full_cond_probs):
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
    state_chain = tuple(data['state'].iloc[-max_chain_len:])

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
            chain_str = " > ".join(lookup_chain)
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
    trade_signal):
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
    latest_state = data['state'].iloc[-1] if 'state' in data.columns else 'N/A'

    report_lines.append(f"- Latest Price Close      : {latest_close:,.2f} USD")
    report_lines.append(f"- Latest Return State     : {latest_state}")
    report_lines.append("\n>>> Continuation Probability Lookup")

    # continuation_result = continuation_probability_lookup(data, conditional_probs)
    #
    # report_lines.append(continuation_result['message'])

    # === CONTINUATION LOOKUP (FULL) ===
    # conditional_probability_lookup_full(
    #     conditional_probs=conditional_probs,
    #     data=data,
    #     report_lines=report_lines
    # )

    # === FULL CONDITIONAL PROBABILITIES ===

    # You pass the current dataset and your full_cond_probs dictionary here!
    full_lookup_result = full_conditional_probability_lookup_full(data, full_cond_probs)
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
    report_lines.append(f"Z-Score       : {runs.get('z_score', 'N/A'):.2f}")

    # Conditional chains
    chains = conditional_probs.get('cond_probs', {}).get(5, {})
    chain_items = list(chains.items())
    n = 3
    report_lines.append(f"\n>>> Top/Tail {n} Conditional Chains (Chain Length: 5)")
    for chain, prob in chain_items[:n]:
        chain_str = " > ".join(chain)
        report_lines.append(f"{chain_str} => Continuation: {prob*100:.2f}%")

    # ... elipsis 
    if len(chains) > n:
        report_lines.append("...")

    for chain, prob in chain_items[-n:]:
        chain_str = " > ".join(chain)
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
