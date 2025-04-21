# .src/risk_explorer.py

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_fetcher_agnostic import get_market_data
from src.statistical_analysis import run_statistical_analysis
from src.filters import adaptive_outlier_filter


def risk_explorer_mode():
    print("\n=== Running Risk Explorer Mode ===\n")

    #Load Config
    with open('config/config.json') as f:
        settings = json.load(f)

    # symbol = settings.get("run_symbol", {}).get("symbol", "CL=F")
    symbols = settings.get("scan_symbols", ["CL=F", "GC=F", "SI=F"])
    start_date = settings.get("run_symbol", {}).get("default_start_date", "2023-01-01")
    end_date = settings.get("run_symbol", {}).get("default_end_date", datetime.today().strftime("%Y-%m-%d"))
    source = settings.get("run_symbol", {}).get("source", "yfinance")
    sample_candles = 2500  # Target number of rows

    timeframes = ["1d", "1h", "15m", "5m", "1m"]


    for symbol in symbols:
        print(f"\n=== Analyzing Symbol: {symbol} ===\n")

        report_lines = []
        report_lines.append("="*50)
        report_lines.append(f"  RISK EXPLORER REPORT - {symbol}")
        report_lines.append("="*50)

        for interval in timeframes:
            print(f"[Info] Fetching {interval} data for {symbol}...")

            if interval in ["1m", "5m", "15m", "30m", "1h"]:
                # Go back enough days (simple assumption)
                days_back = int(sample_candles / (390 if interval == "1m" else 78 if interval == "5m" else 26))
                start_dt = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            else:
                start_dt = start_date

            try:
                raw_data = get_market_data(
                    source=source,
                    symbol=symbol,
                    start_date=start_dt,
                    end_date=end_date,
                    interval=interval
                )
                
                if raw_data.empty:
                    print(f"[Error] No data returned for symbol '{symbol}' in the specified date range.")
                    continue    

                if len(raw_data) > sample_candles:
                    raw_data = raw_data.tail(sample_candles)

                # Run basic processing
                raw_data = run_statistical_analysis(raw_data)

                # Outlier detection
                raw_data = adaptive_outlier_filter(
                    raw_data,
                    column="daily_return_%",
                    method="zscore",
                    z_thresh=3.0
                )

                mean_return = raw_data['daily_return_%'].mean()
                std_return = raw_data['daily_return_%'].std()
                mean_atr = raw_data['atr_14'].mean()

                report_lines.append(f"\nTimeframe: {interval.upper()}")
                report_lines.append(f"- Mean Return       : {mean_return:.4f} %")
                report_lines.append(f"- Std Dev Return    : {std_return:.4f} %")
                report_lines.append(f"- Mean ATR (14)     : {mean_atr:.4f}")

                report_lines.append(f"- Sample Size       : {len(raw_data)} candles")

            except Exception as e:
                print(f"[Error] Failed to analyze {symbol} {interval}: {e}")
                continue

        report_lines.append("\nReport Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        report_content = "\n".join(report_lines)

        print("\n" + report_content)

        # Save
        output_file = f"reports/{symbol.replace('=','_')}_risk_explorer.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n[Info] Risk Explorer report saved to {output_file}\n")
