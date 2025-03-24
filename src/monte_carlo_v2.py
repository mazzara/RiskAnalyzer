# ./OilAnalyzer/monte_carlo_simulation_v2.py

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

    for state in regime_states:
        subset = regimes_df[regimes_df['hmm_state'] == state]
        mu = subset['daily_return_%'].mean() / 100  # to decimal
        sigma = subset['daily_return_%'].std() / 100
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

    print(f"[Risk] VaR ({confidence_level * 100:.0f}%): {var * 100:.2f}%")
    print(f"[Risk] CVaR ({confidence_level * 100:.0f}%): {cvar * 100:.2f}%")

    return var, cvar


# end of monte_carlo_simulation_v2.py
