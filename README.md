# RiskAnalyzer

RiskAnalyzer is a Python-based quantitative market analysis and risk assessment tool. It processes historical financial data to provide **statistical insights**, **conditional probabilities**, **volatility analysis**, and **actionable trade signals**. Inspired by behavioral finance and quantitative methods, it combines techniques like Monte Carlo simulations, GARCH models, and Markov Chain analysis.

---

## ✨ Features

- 📈 **Market Data Fetching**
  - Uses Yahoo Finance via `yfinance` to download OHLCV data.
  - Supports multiple timeframes (1d, 1h, intraday).

- 📊 **Descriptive & Statistical Analysis**
  - Daily returns, volatility, skewness, kurtosis.
  - Adaptive outlier filtering.

- 🔗 **Conditional Probabilities & Continuation Chains**
  - Computes historical state chains (Bullish, Bearish, Neutral).
  - Continuation and reversal probabilities.
  - Markov Chain Transition Probability Matrix (TPM).

- ⚡ **Volatility Models**
  - GARCH model fitting and forecasting.
  - Hidden Markov Models (HMM) for volatility regimes.

- 🎲 **Monte Carlo Simulations**
  - Simple and regime-aware simulations.
  - Value at Risk (VaR) and Conditional VaR (CVaR) calculations.

- 🚀 **Trading Signals**
  - Regime-based strategy (Momentum / Mean Reversion).
  - Entry/Exit price suggestions, Stop Loss, Take Profit.

- 📄 **Detailed Reports**
  - Auto-generated TXT reports.
  - Continuation probability chain snapshots.
  - Volatility and risk metrics.

---

## 📂 Project Structure
```text
RiskAnalyzer/
├── main.py
├── src/
│   ├── cli.py
│   ├── data_fetcher.py
│   ├── descriptive_analysis.py
│   ├── distribution_analysis.py
│   ├── filters.py
│   ├── label_candlesticks.py
│   ├── monte_carlo.py
│   ├── monte_carlo_v2.py
│   ├── conditional_probabilities.py
│   ├── report_generator.py
│   ├── statistical_analysis.py
│   ├── trading_rules.py
│   ├── volatility_analysis.py
│   ├── volatility_regimes.py
│   └── __init__.py
├── reports/
├── data_raw.csv (generated)
├── data_filtered.csv (generated)
├── data_with_stats.csv (generated)
├── data_excluded_outliers.csv (generated)
├── .gitignore
└── README.md 
```

---

## ✅ How to Run

### 1. Install Requirements---
```
pip install -r requirements.txt
```

*Main dependencies include:*
- `pandas`
- `numpy`
- `yfinance`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `hmmlearn`

### 2. Run the Analyzer
run the main.py file
```bash
`python main.py`
```

### 3. Follow CLI Prompts
```bash
=== Risk Analyzer CLI ===

Enter symbol to analyze [default CL=F]: 
Enter timeframe (1d / 1h / 1m / etc): 
Enter end date (YYYY-MM-DD):
```

### 4. Review the Report
- Auto-saves to: `reports/`
- Example: `reports/CL_F_1d_2025-03-24_report.txt`

---

## 🧠 Concepts Used

- **Statistical Analysis**: Descriptive stats, Z-score filtering.
- **Behavioral Finance**: Continuation probabilities, cognitive market perspectives.
- **Monte Carlo Methods**: Simulating price paths for risk metrics.
- **Volatility Forecasting**: GARCH models, regime switching via HMM.
- **Markov Chain Models**: Transition probabilities and market state prediction.
- **Trading Strategy Design**: Bias-driven actions (Momentum / Mean Reversion).

---

## 🛠️ Roadmap Ideas

- [ ] Integrate Real-time Data Streams.
- [ ] Add More Volatility Models (e.g., EGARCH, FIGARCH).
- [ ] Refine Trade Execution Module.
- [ ] Export Report to PDF or HTML.
- [ ] Web Interface (Flask/FastAPI).

---

## 👨‍💻 Author

*Designed and built by [Marcelo Mazzariol](https://github.com/mazzara)*

---

## 📜 License

MIT License  
© 2025 Marcelo Mazzariol

Permission is hereby granted, free of charge, to any person obtaining a copy of this software...

