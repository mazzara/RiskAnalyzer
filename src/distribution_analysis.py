# ./src/distribution_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner outputs
import warnings
warnings.filterwarnings('ignore')

def fit_distribution(data, dist_name):
    """
    Fit a distribution to the data using MLE and return the distribution object and parameters.
    """
    if dist_name == 'normal':
        dist = stats.norm
    elif dist_name == 'lognormal':
        dist = stats.lognorm
    elif dist_name == 'student_t':
        dist = stats.t
    else:
        raise ValueError(f"Distribution '{dist_name}' not supported.")
    
    params = dist.fit(data)
    return dist, params

def plot_distribution_fit(data, dist, params, dist_name, column):
    """
    Plot histogram with fitted PDF and Empirical CDF vs Fitted CDF.
    """
    x = np.linspace(min(data), max(data), 1000)
    
    # PDF
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=30, kde=False, stat='density', label='Empirical', color='skyblue')
    plt.plot(x, dist.pdf(x, *params), 'r-', label=f'{dist_name} PDF')
    plt.title(f"{column}: Histogram with {dist_name} PDF")
    plt.legend()

    # CDF
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, ecdf, marker='.', linestyle='none', label='Empirical CDF')
    plt.plot(x, dist.cdf(x, *params), 'r-', label=f'{dist_name} CDF')
    plt.title(f"{column}: Empirical vs {dist_name} CDF")
    plt.legend()

    plt.tight_layout()
    plt.show()

def goodness_of_fit(data, dist, params):
    """
    Perform Goodness-of-Fit tests: Chi-Square and Anderson-Darling.
    """
    # Chi-Square Test
    observed_freq, bins = np.histogram(data, bins='auto')
    expected_freq = len(data) * np.diff(dist.cdf(bins, *params))
    chi_square_stat = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()

    # Anderson-Darling Test
    ad_test = stats.anderson(data, dist='norm')  # Adjust as necessary for non-normal

    print(f"\n[GoF] Chi-Square Statistic: {chi_square_stat}")
    print(f"[GoF] Anderson-Darling Statistic: {ad_test.statistic}")

    return {
        'chi_square': chi_square_stat,
        'anderson_darling': ad_test.statistic
    }

def probability_of_move(dist, params, threshold, tail='both'):
    """
    Calculate the probability of a move greater than a threshold.
    """
    if tail == 'both':
        prob = dist.sf(threshold, *params) + dist.cdf(-threshold, *params)
    elif tail == 'upper':
        prob = dist.sf(threshold, *params)
    elif tail == 'lower':
        prob = dist.cdf(-threshold, *params)
    else:
        raise ValueError("tail must be 'both', 'upper', or 'lower'.")

    print(f"\n[Probability] Probability of move beyond ±{threshold}: {prob:.4f}")
    return prob

# default test thresholds = 2  or 2% move
def distribution_analysis_pipeline(df, column, thresholds=[2.00]):
    """
    Run distribution fitting, GoF tests, and probability estimation on a given column.
    """
    print(f"\n[Info] Starting Distribution Analysis for {column}...")
    
    data = df[column].dropna()
    results = {}

    # Fit and analyze each distribution
    for dist_name in ['normal', 'lognormal', 'student_t']:
        print(f"\n--- Fitting {dist_name} distribution ---")

        # Fit
        dist, params = fit_distribution(data, dist_name)

        # Plot PDF and CDF
        plot_distribution_fit(data, dist, params, dist_name, column)

        # GoF Tests
        gof_results = goodness_of_fit(data, dist, params)

        # Probability estimates
        prob_results = {}
        for threshold in thresholds:
            prob = probability_of_move(dist, params, threshold)
            prob_results[f'±{threshold}'] = prob

        # Store results
        results[dist_name] = {
            'params': params,
            'gof': gof_results,
            'probabilities': prob_results
        }

    print(f"\n[Info] Distribution Analysis Completed for {column}!")
    return results

