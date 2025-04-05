# .src/scanner.py 

import sys
import os
import json
import datetime as datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import run_analysis
import pandas as pd
from src.cli import prompt_user_inputs, calculate_start_date
from tabulate import tabulate


def load_config():
    with open('config/config.json') as f:
        return json.load(f)


def scanner_mode():
    settings = load_config()
    symbol_list = settings.get("scan_symbols", [])
    default_timeframe = settings.get("default_timeframe", "1d")
    end_date = pd.to_datetime("today").normalize()

    print("\n=== SCANNER MODE ===")
    print(f"Scanning symbols: {', '.join(symbol_list)}")
    print(f"Timeframe: {default_timeframe}")
    print(f"End Date: {end_date.date()}")

    summary_data = []

    for symbol in symbol_list:
        print(f"\n--- Running analysis for {symbol} ---")
        config = prompt_user_inputs(
            symbol=symbol,
            timeframe=default_timeframe,
            end_date=end_date,
            confirm_required=False
        )

        try:
            trade_signal = run_analysis(config)
            summary_data.append({
                'Symbol': symbol,
                'Regime': trade_signal.get('regime', 'N/A'),
                'Bias': trade_signal.get('bias', 'N/A'),
                'Action': trade_signal.get('trade_action', 'N/A'),
                'Confidence': trade_signal.get('confidence', 'N/A'),
                'Position Size': trade_signal.get('position_size', 'N/A'),
                'Entry Price': "{:.2f}".format(trade_signal.get('entry_price', 'N/A')),
                'Stop Loss': "{:.2f}".format(trade_signal.get('stop_loss_price', 'N/A')),
                'Take Profit': "{:.2f}".format(trade_signal.get('take_profit_price', 'N/A'))
            })
        except Exception as e:
            print(f"[ERROR] Skipping {symbol} due to exception: {e}")

    print("\n=== SCAN SUMMARY ===")
    if summary_data:
        print(tabulate(summary_data, headers="keys", tablefmt="pretty"))
    else:
        print("No data to display.")


if __name__ == "__main__":
    scanner_mode()



# end of scanner.py
