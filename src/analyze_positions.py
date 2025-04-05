# .src/analyze_positions.py
import sys 
import os 
import json 
# import datetime as datetime
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import run_analysis
import pandas as pd 
from src.cli import prompt_user_inputs, calculate_start_date 
from tabulate import tabulate


def load_positions():
    filename = 'positions/positions.json'
    if not os.path.exists(filename):
        print(f"[Error] Positions file not found: {filename}")
        return []
    with open(filename) as f:
        positions = json.load(f)
        return positions


def tabulate_summary_results(results):
    if not results:
        print("\n[Info] No results to summarize.")
        return

    print("\n=== POSITION ANALYSIS SUMMARY ===")
    headers = [
        "symbol", 
        "side", 
        "position_size", 
        "entry_price", 
        "current_price",
        "market_value",
        "pnl",
        "pnl_pct",
        # "stop_loss", 
        # "take_profit", 
        # "stop_risk",
        "var_95", 
        "cvar_95"
        # "volatility_regime", 
        # "strategy_bias", 
        # "environment_action",
        # "confidence", 
        # "recommended_exit"
    ]

    rows = [[
        r.get("symbol"),
        r.get("side", "").upper(),
        r.get("position_size"),
        f"{r.get('entry_price'):.2f}",
        f"{r.get('current_price'):.2f}",
        f"{r.get('market_value', 0):.2f}",
        f"{r.get('pnl', 0):.2f}",
        f"{r.get('pnl_percent', 0):.2f}",
        # f"{r.get('stop_loss'):.2f}" if r.get("stop_loss") else "",
        # f"{r.get('take_profit'):.2f}" if r.get("take_profit") else "",
        # f"{r.get('stop_risk', 0):.2f}" if r.get("stop_risk") is not None else "",
        f"{r.get('var_95', 0):.2f}" if r.get("var_95") is not None else "",
        f"{r.get('cvar_95', 0):.2f}" if r.get("cvar_95") is not None else "",
        # r.get("volatility_regime"),
        # r.get("strategy_bias"),
        # r.get("environment_action"),
        # r.get("confidence"),
        # r.get("recommended_exit")
    ] for r in results]

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def analyze_positions():
    print("\n=== POSITION ANALYZER ===")
    positions = load_positions()
    summary_results = []

    if not positions:
        print("[Warning] No positions found in config/positions.json.")
        return

    for pos in positions:
        symbol = pos.get('symbol')
        timeframe = pos.get('timeframe', '1d')
        entry_date_str = pos.get("entry_date")
        entry_date = pd.to_datetime(entry_date_str)
        entry_price = pos.get('entry_price')
        side = pos.get('side', 'long').lower()

        print(f"\n[Position] {symbol} ({side.upper()}) at {entry_price}")

        # Start analysis 90 days before entry
        start_date = (entry_date - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")

        config = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "position_side": side,
            "entry_price": entry_price
        }

        try:
            result = run_analysis(config, position=pos)
            if result:
                summary_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed analysis for {symbol}: {e}")

    tabulate_summary_results(summary_results)



if __name__ == "__main__":
    analyze_positions()

