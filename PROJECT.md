# This project of Risk Analysis of Price - Statistical & Probabilistic Analysis

This project aims to statistically and probabilistically analyze oil price movements, delivering actionable insights for informed trading decisions. It can be expanded and applyed to other financial instruments or commodities.

---

## 📌 Objectives

- Understand statistical properties of oil price movements.
- Evaluate probabilities related to daily movements, volatility, and trend continuations.
- Integrate probabilistic models into a Python-based trading application.

---

## ✅ Step-by-Step Analysis Plan

### 1. Data Preparation & Feature Engineering

Obtain and preprocess historical daily OHLC (Open, High, Low, Close) oil data. Compute the following:

- **Daily Returns (%)**
  \[
  \text{Daily Return} = \frac{Close_t - Close_{t-1}}{Close_{t-1}} \times 100
  \]

- **Daily Range (%)**
  \[
  \text{Range} = \frac{High - Low}{Low} \times 100
  \]

- **True Range (TR) & Average True Range (ATR)**
  \[
  TR_t = \max(High_t - Low_t,\ |High_t - Close_{t-1}|,\ |Low_t - Close_{t-1}|)
  \]

- **Gain/Loss Chains**:
  - Number of consecutive positive (bullish) or negative (bearish) days.

---

### 2. Descriptive Statistical Analysis

Perform initial exploratory analysis to assess:

- **Central Tendency & Dispersion**:
  - Mean, median, mode
  - Standard deviation
  - Skewness and kurtosis

- **Normality Tests**:
  - Shapiro-Wilk Test, Kolmogorov-Smirnov Test
  - QQ-plots for visual inspection

---

### 3. Distribution Analysis & Probabilistic Modeling

Identify appropriate statistical distributions:

- **Candidate Distributions**:
  - Normal distribution
  - Lognormal distribution
  - Student-t distribution
  - Empirical distributions (historical simulation)

- **Fit distributions using**:
  - Maximum Likelihood Estimation (MLE)
  - Goodness-of-fit tests (Chi-square, Anderson-Darling test)

Answer questions such as:

- "What's the probability of oil moving more than ±2% daily?"
- "What daily volatility range is statistically typical?"

---

### 4. Conditional Probability & Continuation Patterns

Analyze runs (continuation/reversal patterns):

- **Runs Test**: Probability of continuation vs. reversal after price moves.
- **Conditional Probability Distributions**: Probability after N consecutive moves.
- **Markov Chain Analysis**:
  - Define states (Bullish, Bearish, Neutral).
  - Calculate Transition Probability Matrix (TPM):

| Current / Next | Bullish | Bearish | Neutral |
|----------------|---------|---------|---------|
| Bullish        | P(B|B)  | P(R|B)  | P(N|B)  |
| Bearish        | P(B|R)  | P(R|R)  | P(N|R)  |
| Neutral        | P(B|N)  | P(R|N)  | P(N|N)  |

---

### 5. Volatility Clustering Analysis (GARCH Models)

Analyze volatility clusters typical in commodity prices:

- **GARCH Model**:
  - Capture volatility clusters, forecast volatility changes.
- **Hidden Markov Model (HMM)**:
  - Identify volatility regimes (low/high volatility periods).
  - Model regime-switching probabilities.

---

### 6. Monte Carlo Simulations

Simulate future price paths using fitted volatility models:

- Generate numerous simulations based on historical data.
- Evaluate probabilistic outcomes:
  - Probability of hitting specific price targets.
  - Worst-case scenarios (Value at Risk – VaR).

---

### 7. Actionable Trading Insights

Translate statistical insights into practical trading decisions:

- Define volatility-based stop-loss and take-profit rules.
- Develop strategies based on conditional probability analysis.
- Generate probabilistically informed trade signals.

---

## 📚 Recommended Models and Studies

- **GARCH Models**: For volatility forecasting.
- **Markov Chains & Hidden Markov Models**: State-transition analysis.
- **Runs Tests & Conditional Probabilities**: Continuation/reversal analysis.
- **Monte Carlo Simulations**: Scenario and risk management.

---

## 🐍 Python Implementation

### Libraries & Tools:

- `pandas`: Data processing.
- `numpy`: Numerical computations.
- `scipy.stats`: Statistical tests and distribution fitting.
- `statsmodels`: Statistical models, GARCH implementations.
- `arch`: Advanced GARCH modeling.
- `matplotlib` & `seaborn`: Visualizations.
- `pyMC` & `hmmlearn`: Hidden Markov Model implementations.

---

## 🚧 Integration into Python Application

Extend current symbol-correlation-based model by adding dedicated modules:

- Data fetching and preprocessing.
- Statistical and probabilistic analysis pipelines.
- GARCH and volatility regime detection.
- Conditional probability and Markov analysis.

Store analytical outcomes systematically in JSON or structured data formats for continued research and integration with existing tools.

---

## 🗃️ Recommended Project Structure

```text
oil_analysis_project/
├── data_fetcher.py            # Fetch and preprocess oil data
├── statistical_analysis.py    # Compute descriptive statistics
├── distribution_fitting.py    # Fit and test distributions
├── garch_model.py             # Volatility modeling (GARCH)
├── markov_analysis.py         # Transition probability analysis
├── monte_carlo_sim.py         # Scenario simulations
├── visualization.py           # Charts and data visualization
├── trading_rules.py           # Actionable trading strategies
└── main.py                    # Entry-point and pipeline execution

