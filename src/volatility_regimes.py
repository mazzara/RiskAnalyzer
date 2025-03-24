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
def hmm_volatility_pipeline(df, return_column='daily_return_%', n_states=2):
    """
    Full pipeline for fitting HMM, plotting regimes and probabilities.
    """
    print("\n[Info] Starting Hidden Markov Model (HMM) Regime Detection...")

    # Step 1: Fit HMM model
    model, df_with_states = fit_hmm(df, return_column=return_column, n_states=n_states)

    # Step 2: Plot states (scatter by regime)
    plot_hmm_states(df_with_states, return_column=return_column)

    # Step 3: Plot regime probabilities
    plot_state_probabilities(model, df_with_states)

    print("\n[Info] HMM Regime Detection Completed.")
    return model, df_with_states


# end of volatility_regimes.py
