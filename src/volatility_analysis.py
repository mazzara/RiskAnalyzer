# ./src/volatility_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

sns.set(style='whitegrid')

# --------------------------------------------------------
# 1. Fit GARCH(1,1) Model
# --------------------------------------------------------
def fit_garch(df, return_column='daily_return_%', p=1, q=1):
    returns = df[return_column].dropna()/100

    # Rescale returns for intraday (optional but useful)
    # returns_rescaled = returns #* 100

    am = arch_model(returns, p=p, q=q, mean='Zero', vol='GARCH', dist='normal')
    model = am.fit(disp='off')

    if model.convergence_flag != 0:
        print(f"WARNING: GARCH model did not fully converge (code {model.convergence_flag})")

    cond_volatility = model.conditional_volatility #/100

    return model


# --------------------------------------------------------
# 2. Plot Volatility Clusters (Conditional Volatility)
# --------------------------------------------------------
def plot_volatility(df, garch_results, return_column='daily_return_%'):
    # Get forecast volatility
    forecast_vol = garch_results.conditional_volatility

    # Check for time axis column
    time_axis = None
    for col in ['date', 'datetime', 'timestamp']:
        if col in df.columns:
            time_axis = df[col]
            break

    if time_axis is None:
        # Fallback to index
        time_axis = df.index

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, df[return_column], label='Returns')
    plt.title('Returns')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis[-len(forecast_vol):], forecast_vol, color='orange', label='GARCH Volatility')
    plt.title('GARCH Conditional Volatility')
    plt.legend()

    plt.tight_layout()
    plt.show()



# --------------------------------------------------------
# 3. Forecast Volatility
# --------------------------------------------------------
def forecast_volatility(garch_results, steps=5):
    forecasts = garch_results.forecast(horizon=steps)

    vol_forecasts = forecasts.variance.values[-1, :]

    # Convert variance to std dev and rescale from % to decimal
    forecast_vols = np.sqrt(vol_forecasts) #/ 100

    df_forecast = pd.DataFrame({
        'forecast_mean': [0.0]*steps,
        'forecast_volatility': forecast_vols
    })

    print("\n[Info] Volatility Forecast (Decimal % Move):\n", df_forecast.head())
    return df_forecast




# end of volatility_analysis.py
