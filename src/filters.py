# ./OilAnalyzer/filters.py
import pandas as pd
import numpy as np

def adaptive_outlier_filter(
    df,
    column='daily_return_%',
    method='combined',
    z_thresh=3.0,
    lower_quantile=0.01,
    upper_quantile=0.99,
    audit_filename='excluded_outliers.csv'
):
    """
    Adaptive outlier filter supporting z-score, quantile, or both.
    
    Parameters:
    - df: DataFrame containing data
    - column: column to filter (default = 'daily_return_%')
    - method: 'zscore', 'quantile', or 'combined' (default)
    - z_thresh: threshold for z-score method (default = 3.0)
    - lower_quantile: lower threshold for quantile method (default = 0.01)
    - upper_quantile: upper threshold for quantile method (default = 0.99)
    - audit_filename: filename to save excluded outliers

    Returns:
    - df_filtered: cleaned DataFrame
    """
    df = df.copy()
    
    # Start with no outlier flags
    df['is_outlier'] = False
    
    # Z-SCORE FILTERING
    if method in ['zscore', 'combined']:
        mean_val = df[column].mean()
        std_val = df[column].std()

        df['z_score'] = (df[column] - mean_val) / std_val

        # Mark z-score outliers
        df.loc[df['z_score'].abs() > z_thresh, 'is_outlier'] = True
        
        print(f"[Filter] Z-Score outlier filter applied: Threshold = {z_thresh}")

    # QUANTILE FILTERING
    if method in ['quantile', 'combined']:
        lower_bound = df[column].quantile(lower_quantile)
        upper_bound = df[column].quantile(upper_quantile)

        # Mark quantile outliers
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), 'is_outlier'] = True
        
        print(f"[Filter] Quantile outlier filter applied: "
              f"Lower {lower_quantile*100:.1f}%, Upper {upper_quantile*100:.1f}%")

    # SAVE EXCLUDED OUTLIERS
    outliers = df[df['is_outlier']].copy()
    if not outliers.empty:
        outliers.to_csv(audit_filename, index=False)
        print(f"[Audit] {len(outliers)} outlier rows saved to '{audit_filename}'")

    # CLEAN DATAFRAME
    df_filtered = df[~df['is_outlier']].copy()

    # Drop temporary columns
    df_filtered.drop(columns=['is_outlier'], inplace=True)
    if 'z_score' in df_filtered.columns:
        df_filtered.drop(columns=['z_score'], inplace=True)

    print(f"[Filter] Cleaned dataset: {len(df_filtered)} rows (removed {len(outliers)} outliers)")

    return df_filtered

# end of filters.py
