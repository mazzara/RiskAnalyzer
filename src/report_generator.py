# ./OilAnalyser/report_generator.py

import os
from datetime import datetime

# Ensure reports folder exists
REPORTS_FOLDER = 'reports'
os.makedirs(REPORTS_FOLDER, exist_ok=True)


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
    report_lines.append(f"{'OIL ANALYZER FINAL REPORT':^50}")
    report_lines.append("=" * 50)
    report_lines.append(f"Symbol        : {symbol}")
    report_lines.append(f"Timeframe     : {timeframe}")
    report_lines.append(f"Date Range    : {start_date} to {end_date}")
    report_lines.append(f"Data Points   : {len(data)}")
    report_lines.append("-" * 50)

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
    for chain, prob in chains.items():
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
        report_lines.append(f"VaR (95%): {v1.get('var', 'N/A'):.2f} %")
        report_lines.append(f"CVaR (95%): {v1.get('cvar', 'N/A'):.2f} %")

        report_lines.append("\n-- Regime & Volatility Aware Monte Carlo (v2):")
        report_lines.append(f"VaR (95%): {v2.get('var', 'N/A'):.2f} %")
        report_lines.append(f"CVaR (95%): {v2.get('cvar', 'N/A'):.2f} %")

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
    report_content = "\n".join(report_lines)

    # === PRINT TO CONSOLE ===
    print("\n" + report_content + "\n")

    # === SAVE TO FILE ===
    file_name = f"{symbol.replace('=','_')}_{timeframe}_{end_date}_report.txt"
    file_path = os.path.join(REPORTS_FOLDER, file_name)

    with open(file_path, 'w') as file:
        file.write(report_content)

    print(f"[Info] Report saved to {file_path}")

# end of report_generator.py
