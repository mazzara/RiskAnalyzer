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
