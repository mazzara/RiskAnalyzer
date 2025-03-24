# ./RiskAnalyzer/main.py  
from src.cli import prompt_user_inputs
from src.data_fetcher import get_market_data 
from src.statistical_analysis import run_statistical_analysis
from src.filters import adaptive_outlier_filter
from src.descriptive_analysis import descriptive_statistics_pipeline
from src.distribution_analysis import distribution_analysis_pipeline
from src.conditional_probabilities import conditional_probabilities_pipeline, full_conditional_probabilities
from src.volatility_analysis import fit_garch, plot_volatility, forecast_volatility
from src.volatility_regimes import hmm_volatility_pipeline
from src.monte_carlo import monte_carlo_simulation, plot_simulations, calculate_var_cvar
from src.monte_carlo_v2 import monte_carlo_simulation_v2, plot_simulations, calculate_var_cvar
from src.trading_rules import determine_regime_strategy, determine_trade_signal
from src.label_candlesticks import classify_candle
from src.report_generator import generate_report, full_conditional_probability_lookup_full, full_conditional_probability_lookup_full_verbose


if __name__ == '__main__':
    # symbol = 'CL=F'
    # start = '2025-02-01'
    # end = '2025-03-21'
    # interval = '30m'

    # CLI interaction
    config = prompt_user_inputs()

    if config is None:
        exit()

    symbol = config['symbol']
    start = config['start_date']
    end = config['end_date']
    interval = config['timeframe']

    print(f"\n[Info] Fetching data for {symbol} from {start} to {end}, timeframe: {interval}")


    # ==== Step 1: Fetch data
    raw_data = get_market_data(symbol, start, end, interval)
    if raw_data.empty:
        print("[Error] No data fetched. Exiting.")
        exit()

    print(raw_data.head(), "\n", raw_data.tail())

    # Step 2: Rename columns
    # raw_data.rename(columns={
    #     'open_cl=f': 'open',
    #     'high_cl=f': 'high',
    #     'low_cl=f': 'low',
    #     'close_cl=f': 'close',
    #     'adj_close_cl=f': 'adj_close',
    #     'volume_cl=f': 'volume'
    # }, inplace=True)

    # ==== Step 3: Run statistical analysis
    raw_data = run_statistical_analysis(raw_data)
    raw_data.to_csv('data_raw.csv', index=False)

    # Step 3.1: Filter outliers using combined method (Z-score + Quantile)
    data_filtered = adaptive_outlier_filter(
        raw_data,
        column='daily_return_%',
        method='combined',
        z_thresh=3.0,
        lower_quantile=0.01,
        upper_quantile=0.99,
        audit_filename='data_excluded_outliers.csv'
    )
    data_filtered.to_csv('data_filtered.csv', index=False)

    data = data_filtered.copy()

    # ==== Step 4: Label Candlesticks
    data['candlestick'] = data.apply(
        lambda row: classify_candle(row['open'], row['high'], row['low'], row['close']),
        axis=1
    )

    # ==== Step 5: Descriptive stats & distribution analysis
    columns_to_analyze = ['daily_return_%', 'daily_range_%', 'tr', 'atr_14']

    # Capture Returns
    desc_stats = descriptive_statistics_pipeline(data, columns_to_analyze)
    dist_results = distribution_analysis_pipeline(data, column='daily_return_%', thresholds=[1,2,3])

    # ==== Step 6: Conditional Probabilities & Continuation Patterns
    cond_probs_results = conditional_probabilities_pipeline(data, threshold=0.0, max_chain=5)

    # NEW: Full Conditional Probabilities for All States
    full_cond_probs = full_conditional_probabilities(data, max_chain=5)
    print(full_cond_probs.keys())
    print("[Debug] Example for chain length 5:", full_cond_probs.get(5, {}))

    # After conditional_probs_results is generated...
    print("\n[Verbose] Conditional Probability Chain Analysis...")

    # Current chain snapshot (up to max_chain)
    max_chain_len = max(full_cond_probs.keys(), default=5)
    current_chain = tuple(data['state'].iloc[-max_chain_len:])

    print(f"[Verbose] Current Chain (latest {max_chain_len} states): {' > '.join(current_chain)}")

    # Run the lookup function (reusing the function from report_generator.py)
    probability, next_state_probs = full_conditional_probability_lookup_full_verbose(data, full_cond_probs)

    # Display results
    print(f"\n[Verbose] Continuation Probability for Chain '{' > '.join(current_chain)}':")
    if probability is not None:
        print(f"- Continuation Probability (stay {current_chain[-1]}): {probability * 100:.2f}%")
    else:
        print("- No continuation probability available for this chain.")

    print("\n[Verbose] Probabilities for Next State (Chain Lookup):")
    if next_state_probs:
        for state, prob in next_state_probs.items():
            direction = "continuation" if state == current_chain[-1] else "reversal"
            print(f"- {state} ({direction}): {prob * 100:.2f}%")
    else:
        print("- No next state probabilities available for this chain.")

    # ==== Step 7: Volatility Analysis
    garch_results = fit_garch(data, return_column='daily_return_%')
    plot_volatility(data, garch_results)
    forecast_df = forecast_volatility(garch_results, steps=5)
    print(forecast_df)

    # ==== Step 8: Run HMM regime detection
    hmm_model, data_with_regimes = hmm_volatility_pipeline(data, return_column='daily_return_%', n_states=2)

    # ==== Step 9: Monte Carlo Simultion - v1.0 (Simple)
    # Get current price
    current_price = data['close'].iloc[-1]
    # Simple mu/sigma estimates from historical data
    mu = data['daily_return_%'].mean()/100
    sigma = data['daily_return_%'].std()/100
    print(f"Current Prices: {current_price}, Mu: {mu:.5f}, Sigma: {sigma:.5f}")

    n_steps = 10
    n_simulations = 500

    simulated_price_paths = monte_carlo_simulation(
            start_price=current_price,
            mu=mu,
            sigma=sigma,
            n_steps=n_steps,
            n_simulations=n_simulations)
    # Plot simulated price paths
    plot_simulations(simulated_price_paths)
    # Calculate VaR and CVaR
    var_v1, cvar_v1 = calculate_var_cvar(simulated_price_paths, confidence_level=0.95)


    # ==== Step 10: Monte carlo v2.0 (Regime + Volatility Aware)
    current_price = data['close'].iloc[-1]
    print(f"Current Price: {current_price}")

    n_steps = 10
    n_simulations = 500

    simulated_price_paths = monte_carlo_simulation_v2(
        start_price=current_price,
        regimes_df=data_with_regimes,
        garch_results=garch_results,
        hmm_model=hmm_model,
        n_steps=n_steps,
        n_simulations=n_simulations
    )

    # Plot simulations
    plot_simulations(simulated_price_paths)

    # Risk metrics
    var_v2, cvar_v2 = calculate_var_cvar(simulated_price_paths, confidence_level=0.95)


    # ==== Step 11: Actionable Trading Insights
    current_regime = data_with_regimes['hmm_state'].iloc[-1]
    next_vol_forecast = forecast_volatility(garch_results, steps=1).iloc[0]['forecast_volatility']
    continuation_prob = 0.65  # Example from Markov or Chains
    reversal_prob = 0.35      # Example from Markov or Chains
    current_price = data['close'].iloc[-1]
    mean_price = data['close'].rolling(window=20).mean().iloc[-1]  # 20-SMA

    # Generate trade signal
    trade_signal = determine_trade_signal(
        regime=current_regime,
        current_price=current_price,
        mean_price=mean_price,
        continuation_prob=continuation_prob,
        reversal_prob=reversal_prob,
        garch_vol=next_vol_forecast,
        account_size=50000,
        risk_per_trade=0.01
    )

    # Display output
    print("\n[Trade Signal]")
    for k, v in trade_signal.items():
        print(f"{k}: {v}")

    # Show and/or save
    print(data.head(20))
    data.to_csv('data_with_stats.csv', index=False)

    # Final step === Generate Report ===
    generate_report(
        symbol=symbol,
        timeframe=interval,
        start_date=start,
        end_date=end,
        data=data,
        descriptive_stats=desc_stats,
        distribution_results=dist_results,
        conditional_probs=cond_probs_results,
        full_cond_probs=full_cond_probs,
        garch_results=garch_results,  # fine, keep if you want HMM link
        volatility_forecast=forecast_df,  # <-- pass this explicitly!
        hmm_results=hmm_model,
        monte_carlo={
            'v1': {'var': var_v1, 'cvar': cvar_v1},
            'v2': {'var': var_v2, 'cvar': cvar_v2}
        },
        trade_signal=trade_signal
    )

    print("\n[Info] Analysis Complete.\n")


# end of main.py
