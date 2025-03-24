# ./OilAnalyzer/descriptive_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def compute_central_tendency_dispersion(df, column):
    """
    Compute central tendency and dispersion metrics for a given column
    """
    metrics = {}
    data = df[column].dropna()

    metrics['mean'] = data.mean()
    metrics['median'] = data.median()
    metrics['mode'] = data.mode().iloc[0] if not data.mode().empty else None
    metrics['std_dev'] = data.std()
    metrics['variance'] = data.var()
    metrics['skewness'] = data.skew()
    metrics['kurtosis'] = data.kurtosis()

    print(f"\n[Info] Descriptive Stats for {column}:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics

def perform_normality_tests(df, column):
    """
    Perform normality tests on a given column
    """
    data = df[column].dropna()

    print(f"\n[Info] Normality Tests for {column}:")
    
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk Test: statistic={shapiro_stat}, p-value={shapiro_p}")

    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f"Kolmogorov-Smirnov Test: statistic={ks_stat}, p-value={ks_p}")

    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p
    }

def plot_distribution(df, column):
    """
    Plot histogram, density plot, and QQ-plot for a given column
    """
    data = df[column].dropna()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"Histogram and Density Plot for {column}")

    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"QQ-Plot for {column}")

    plt.tight_layout()
    plt.show()

def descriptive_statistics_pipeline(df, columns_to_analyze):
    """
    Run descriptive analysis pipeline on a list of columns
    """
    print("[Info] Starting Descriptive Statistical Analysis...")
    results = {}

    for column in columns_to_analyze:
        print(f"\n--- Analyzing column: {column} ---")
        summary_stats = compute_central_tendency_dispersion(df, column)
        normality_tests = perform_normality_tests(df, column)
        plot_distribution(df, column)

        results[column] = {
            'summary': summary_stats,
            'normality': normality_tests
        }

    print("[Info] Descriptive Statistical Analysis Completed!")
    return results

