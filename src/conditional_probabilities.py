# ./src/conditional_probabilities.py

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Suppress warnings for cleaner outputs
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. CLASSIFY STATES
# -------------------------------
def classify_state(row, threshold=0.0):
    """
    Classify price movement state: Bullish, Bearish, or Neutral
    Args:
        row: pandas Series (single row with daily return)
        threshold: return threshold to classify as Neutral
    Returns:
        String: 'Bullish', 'Bearish', or 'Neutral'
    """
    if row['daily_return_%'] > threshold:
        return 'Bullish'
    elif row['daily_return_%'] < -threshold:
        return 'Bearish'
    else:
        return 'Neutral'


def classify_state_dynamic(row, threshold):
    if row['daily_return_%'] > threshold:
        return 'Bullish'
    elif row['daily_return_%'] < -threshold:
        return 'Bearish'
    else:
        return 'Neutral'


def get_dynamic_neutral_threshold(hmm_model):
    """
    Compute a dynamic threshold based on HMM state means.
    Returns:
        threshold: float (in decimal, e.g., 0.0042 for 0.42%)
    """
    hmm_means = hmm_model.means_.flatten()
    return max(abs(hmm_means.min()), abs(hmm_means.max())) / 2


# -------------------------------
# 2. RUNS TEST
# -------------------------------
def runs_test(df, state_column='state'):
    """
    Perform a runs test to check for randomness (trendiness vs mean reversion)
    Args:
        df: DataFrame with a 'state' column
        state_column: column name with classified states (Bullish/Bearish/Neutral)
    Returns:
        runs_info: dict with run counts and expected/observed comparisons
    """
    # Convert states to binary (Bullish=1, Bearish=0), ignore Neutral for now
    binary_seq = df[df[state_column] != 'Neutral'][state_column].map({'Bullish': 1, 'Bearish': 0}).values

    n1 = np.sum(binary_seq)
    n2 = len(binary_seq) - n1
    total_runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1])

    # Expected runs and standard deviation (Wald–Wolfowitz runs test)
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                       (((n1 + n2) ** 2) * (n1 + n2 - 1)))

    z_score = (total_runs - expected_runs) / std_runs

    print(f"\n[Runs Test]")
    print(f"Number of Bullish: {n1}")
    print(f"Number of Bearish: {n2}")
    print(f"Total Runs: {total_runs}")
    print(f"Expected Runs: {expected_runs:.2f}")
    print(f"Z-score: {z_score:.2f}")

    runs_info = {
        'bullish_count': n1,
        'bearish_count': n2,
        'total_runs': total_runs,
        'expected_runs': expected_runs,
        'z_score': z_score
    }

    return runs_info

# -------------------------------
# 3. CONDITIONAL PROBABILITY DISTRIBUTIONS
# -------------------------------
def conditional_probabilities(df, state_column='state', max_chain=5):
    """
    Compute conditional probabilities of continuation/reversal given N previous states
    Args:
        df: DataFrame with a 'state' column
        max_chain: maximum chain length to consider
    Returns:
        cond_probs: dict of chain length probabilities
    """
    cond_probs = {}
    states = df[state_column].values
    length = len(states)

    for chain_len in range(1, max_chain + 1):
        continuation_counts = Counter()
        total_counts = Counter()

        for i in range(chain_len, length - 1):
            prev_chain = tuple(states[i - chain_len:i])
            next_state = states[i]

            total_counts[prev_chain] += 1
            if prev_chain[-1] == next_state:
                continuation_counts[prev_chain] += 1

        probabilities = {
            chain: continuation_counts[chain] / total_counts[chain]
            for chain in total_counts
        }

        cond_probs[chain_len] = probabilities

        # Print conditional probabilities -- mostly for debugging
        # print(f"\n[Conditional Probabilities] Chain length: {chain_len}")
        # for chain, prob in probabilities.items():
        #     print(f"Prev {chain} => Continuation Probability: {prob:.2f}")

    return cond_probs

# -------------------------------
# 4. MARKOV CHAIN ANALYSIS (TPM)
# -------------------------------
def markov_chain_tpm(df, state_column='state'):
    """
    Calculate Markov Chain Transition Probability Matrix (TPM)
    Args:
        df: DataFrame with a 'state' column
    Returns:
        tpm_df: DataFrame TPM
    """
    states = df[state_column].values
    # unique_states = ['Bullish', 'Bearish', 'Neutral']
    unique_states = df[state_column].dropna().unique().tolist()

    transitions = {state: Counter() for state in unique_states}

    for current, next_ in zip(states[:-1], states[1:]):
        transitions[current][next_] += 1

    tpm = {}
    for state in unique_states:
        total = sum(transitions[state].values())
        tpm[state] = {next_state: (transitions[state][next_state] / total) if total > 0 else 0
                      for next_state in unique_states}

    tpm_df = pd.DataFrame(tpm).T
    print("\n[Transition Probability Matrix (TPM)]")
    print(tpm_df)

    return tpm_df

# -------------------------------
# 5. TPM HEATMAP VISUALIZATION
# -------------------------------
def plot_tpm_heatmap(tpm_df):
    """
    Plot Transition Probability Matrix as a heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(tpm_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title("Transition Probability Matrix (TPM) Heatmap")
    plt.ylabel("Current State")
    plt.xlabel("Next State")
    plt.show()

# -------------------------------
# 6. PIPELINE FUNCTION
# # -------------------------------

def conditional_probabilities_pipeline(df, threshold=0.0, max_chain=5, show_plots=True, state_column='state'):
    print("\n[Info] Starting Conditional Probabilities & Continuation Patterns Analysis...")

    if state_column == 'state':  # Only classify if using default column
        df[state_column] = df.apply(lambda row: classify_state(row, threshold=threshold), axis=1)

    # No filtering unless explicitly needed
    # df = df[df[state_column] != "Neutral"].copy()

    runs_info = {}
    if df[state_column].nunique() == 2:
        runs_info = runs_test(df, state_column=state_column)
    else:
        print(f"[Warning] Runs test skipped: '{state_column}' has {df[state_column].nunique()} unique values (expected binary).")

    cond_probs = conditional_probabilities(df, state_column=state_column, max_chain=max_chain)
    tpm_df = markov_chain_tpm(df, state_column=state_column)

    if show_plots:
        plot_tpm_heatmap(tpm_df)

    return {
        'runs_info': runs_info,
        'cond_probs': cond_probs,
        'tpm': tpm_df
    }


def full_conditional_probabilities(df, state_column='state', max_chain=5):
    """
    Compute full conditional probabilities for all next states 
    given N previous states (chain of length N).

    Returns:
        dict of: {chain_len: {prev_chain: {state: prob}}}
    """
    from collections import Counter

    full_cond_probs = {}
    states = df[state_column].values
    length = len(states)

    for chain_len in range(1, max_chain + 1):
        next_state_counts = {}  # holds {prev_chain: {next_state: count}}
        total_counts = {}

        for i in range(chain_len, length - 1):
            prev_chain = tuple(states[i - chain_len:i])
            next_state = states[i]

            # Initialize if new chain
            if prev_chain not in next_state_counts:
                next_state_counts[prev_chain] = Counter()

            # Count transitions
            next_state_counts[prev_chain][next_state] += 1

            # Track total counts for this chain
            total_counts[prev_chain] = total_counts.get(prev_chain, 0) + 1

        # Convert counts to probabilities
        chain_probs = {}
        for prev_chain, state_counts in next_state_counts.items():
            chain_probs[prev_chain] = {
                state: count / total_counts[prev_chain]
                for state, count in state_counts.items()
            }

        full_cond_probs[chain_len] = chain_probs

    return full_cond_probs

# end of conditional_probabilities.py
