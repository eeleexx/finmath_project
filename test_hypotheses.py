import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from model import FinancialMarket


def calculate_rv(price_series, window=20):
    """Calculate Realized Volatility (Annualized) using rolling standard deviation of log returns."""
    log_returns = np.log(price_series / price_series.shift(1))
    # Annualize by sqrt(252)
    rv = log_returns.rolling(window=window).std() * np.sqrt(252)
    return rv


def test_hypothesis_1_divergence_spread():
    print("\n--- Testing Hypothesis 1: Divergence vs (IV - RV) Spread ---")
    # Scenario A: Noise=VADER, Sophisticated=FinBERT (Default)
    model = FinancialMarket(
        stock_symbol="AMD",
        num_noise=30,
        num_sophisticated=5,
        sentiment_key_title="vader_title",
        sentiment_key_content="finbert_content",
    )

    steps = 252  # 1 Year
    for _ in range(steps):
        if not model.running:
            break
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    # Calculate RV
    df["RV"] = calculate_rv(df["StockPrice"])

    # Calculate Spread (IV_ATM - RV)
    df["Spread"] = df["ImpliedVol_ATM"] - df["RV"]

    # Clean NaN (first 20 days)
    clean_df = df.dropna(subset=["Spread", "Divergence"])

    # Export to CSV
    clean_df.to_csv("results/results_hypothesis_1.csv")
    print("Data saved to results/results_hypothesis_1.csv")

    if len(clean_df) < 10:
        print("Not enough data points for regression.")
        return

    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_df["Divergence"], clean_df["Spread"]
    )

    print(f"Correlation: {r_value:.4f}")
    print(f"P-Value: {p_value:.4e}")
    print(f"Slope: {slope:.4f}")

    if slope > 0 and p_value < 0.05:
        print("RESULT: Hypothesis 1 SUPPORTED (Positive significant correlation).")
    else:
        print("RESULT: Hypothesis 1 NOT supported.")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clean_df, x="Divergence", y="Spread", alpha=0.5)
    sns.regplot(data=clean_df, x="Divergence", y="Spread", scatter=False, color="red")
    plt.title(f"Hypothesis 1: Divergence vs IV-RV Spread (Corr={r_value:.2f})")
    plt.xlabel("Sentiment Divergence |Title(VADER) - Content(FinBERT)|")
    plt.ylabel("Volatility Spread (IV_ATM - RV)")
    plt.savefig("figures/hypothesis_1_result.png")
    print("Plot saved to figures/hypothesis_1_result.png")


def test_hypothesis_2_vol_smile():
    print("\n--- Testing Hypothesis 2: Noise Traders impact on Volatility Smile ---")

    # Scenario A: Low Noise
    print("Running Scenario A: Low Noise (5 agents)...")
    model_low = FinancialMarket(stock_symbol="AMD", num_noise=5, num_sophisticated=10)
    for _ in range(252):
        if not model_low.running:
            break
        model_low.step()
    df_low = model_low.datacollector.get_model_vars_dataframe()
    df_low["Scenario"] = "LowNoise"

    # Scenario B: High Noise
    print("Running Scenario B: High Noise (50 agents)...")
    model_high = FinancialMarket(stock_symbol="AMD", num_noise=50, num_sophisticated=10)
    for _ in range(252):
        if not model_high.running:
            break
        model_high.step()
    df_high = model_high.datacollector.get_model_vars_dataframe()
    df_high["Scenario"] = "HighNoise"

    # Export Data
    combined_df = pd.concat([df_low, df_high])
    combined_df.to_csv("results/results_hypothesis_2_smile.csv")
    print("Data saved to results/results_hypothesis_2_smile.csv")

    # Statistical Test (t-test)
    # Testing if High Noise OTM Call IV is significantly greater than Low Noise
    t_stat, p_val = stats.ttest_ind(
        df_high["ImpliedVol_OTM_Call"], df_low["ImpliedVol_OTM_Call"], equal_var=False
    )
    print(f"t-test (High vs Low Noise OTM IV): t={t_stat:.4f}, p={p_val:.4e}")
    if p_val < 0.05 and t_stat > 0:
        print(
            "RESULT: Difference is Statistically Significant (High Noise > Low Noise)."
        )
    else:
        print("RESULT: Difference is NOT Statistically Significant.")

    # Calculate Average IV Structure at end of simulation
    avg_iv_low = {
        "0.9 (OTM Put)": df_low["ImpliedVol_OTM_Put"].mean(),
        "1.0 (ATM)": df_low["ImpliedVol_ATM"].mean(),
        "1.1 (OTM Call)": df_low["ImpliedVol_OTM_Call"].mean(),
    }

    avg_iv_high = {
        "0.9 (OTM Put)": df_high["ImpliedVol_OTM_Put"].mean(),
        "1.0 (ATM)": df_high["ImpliedVol_ATM"].mean(),
        "1.1 (OTM Call)": df_high["ImpliedVol_OTM_Call"].mean(),
    }

    print("Average IV Structure (Low Noise):", avg_iv_low)
    print("Average IV Structure (High Noise):", avg_iv_high)

    # Plot comparison
    moneyness = [0.9, 1.0, 1.1]
    ivs_low = [
        avg_iv_low["0.9 (OTM Put)"],
        avg_iv_low["1.0 (ATM)"],
        avg_iv_low["1.1 (OTM Call)"],
    ]
    ivs_high = [
        avg_iv_high["0.9 (OTM Put)"],
        avg_iv_high["1.0 (ATM)"],
        avg_iv_high["1.1 (OTM Call)"],
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        moneyness,
        ivs_low,
        marker="o",
        label="Low Noise (Rational Market)",
        linestyle="--",
    )
    plt.plot(
        moneyness,
        ivs_high,
        marker="o",
        label="High Noise (Speculative Market)",
        linewidth=2.5,
    )
    plt.title("Hypothesis 2: Volatility Smile Shape")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("Implied Volatility")
    plt.xticks(moneyness)
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/hypothesis_2_result.png")
    print("Plot saved to figures/hypothesis_2_result.png")


def test_hypothesis_3_mm_hedging():
    print(
        "\n--- Testing Hypothesis 3: MM Delta-Hedging P&L vs Information Asymmetry ---"
    )

    # 1. High Asymmetry / High Noise Scenario
    print("Running High Noise Scenario (50 Noise Traders)...")
    model_high = FinancialMarket(stock_symbol="AMD", num_noise=50, num_sophisticated=5)
    for _ in range(252):
        if not model_high.running:
            break
        model_high.step()

    df_high = model_high.datacollector.get_model_vars_dataframe()
    df_high["Scenario"] = "HighNoise"
    final_wealth_high = df_high["MMPortfolioValue"].iloc[-1]

    # 2. Low Asymmetry / Low Noise Scenario
    print("Running Low Noise Scenario (5 Noise Traders)...")
    model_low = FinancialMarket(
        stock_symbol="AMD", num_noise=5, num_sophisticated=50
    )  # Invert ratio
    for _ in range(252):
        if not model_low.running:
            break
        model_low.step()

    df_low = model_low.datacollector.get_model_vars_dataframe()
    df_low["Scenario"] = "LowNoise"
    final_wealth_low = df_low["MMPortfolioValue"].iloc[-1]

    # Export Data
    combined_df = pd.concat([df_low, df_high])
    combined_df.to_csv("results/results_hypothesis_3_wealth.csv")
    print("Data saved to results/results_hypothesis_3_wealth.csv")

    # Statistical Test (t-test on Daily P&L)
    pnl_high = df_high["MMPortfolioValue"].diff().dropna()
    pnl_low = df_low["MMPortfolioValue"].diff().dropna()

    # We expect Low Noise P&L > High Noise P&L
    t_stat, p_val = stats.ttest_ind(pnl_low, pnl_high, equal_var=False)

    print(f"t-test (Low vs High Noise Daily P&L): t={t_stat:.4f}, p={p_val:.4e}")
    if p_val < 0.05 and t_stat > 0:
        print(
            "RESULT: Difference is Statistically Significant (Low Noise P&L > High Noise P&L)."
        )
    else:
        print("RESULT: Difference is NOT Statistically Significant.")

    print(f"Final MM Wealth (High Noise): ${final_wealth_high:,.2f}")
    print(f"Final MM Wealth (Low Noise):  ${final_wealth_low:,.2f}")

    if final_wealth_low > final_wealth_high:
        print("RESULT: Hypothesis 3 SUPPORTED (MM Wealth Low Noise > High Noise).")
    else:
        print("RESULT: Hypothesis 3 NOT supported.")

    # Plot Wealth Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_high["MMPortfolioValue"], label="High Noise (High Asymmetry)", color="red"
    )
    plt.plot(
        df_low["MMPortfolioValue"], label="Low Noise (Low Asymmetry)", color="green"
    )
    plt.title("Hypothesis 3: MM Wealth Evolution")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/hypothesis_3_result.png")
    print("Plot saved to figures/hypothesis_3_result.png")


def test_sentiment_comparison():
    print("\n--- Testing Sentiment Comparison: VADER vs FinBERT ---")

    # 1. Mixed Model: Noise uses VADER (Title), Sophisticated uses FinBERT (Content)
    print("Running Mixed Model (Noise=VADER, Soph=FinBERT)...")
    model_mixed = FinancialMarket(
        stock_symbol="AMD",
        num_noise=30,
        num_sophisticated=30,
        sentiment_key_title="vader_title",
        sentiment_key_content="finbert_content",
    )
    for _ in range(252):
        if not model_mixed.running:
            break
        model_mixed.step()
    df_mixed = model_mixed.datacollector.get_model_vars_dataframe()
    df_mixed["Model"] = "Mixed"

    # 2. Pure FinBERT Model: Noise uses FinBERT (Title), Sophisticated uses FinBERT (Content)
    print("Running Pure FinBERT Model (Noise=FinBERT, Soph=FinBERT)...")
    model_finbert = FinancialMarket(
        stock_symbol="AMD",
        num_noise=30,
        num_sophisticated=30,
        sentiment_key_title="finbert_title",
        sentiment_key_content="finbert_content",
    )
    for _ in range(252):
        if not model_finbert.running:
            break
        model_finbert.step()
    df_finbert = model_finbert.datacollector.get_model_vars_dataframe()
    df_finbert["Model"] = "FinBERT"

    # Export Data
    combined_df = pd.concat([df_mixed, df_finbert])
    combined_df.to_csv("results/results_sentiment_comparison.csv")
    print("Data saved to results/results_sentiment_comparison.csv")

    # Compare Volatility of Stock Price
    vol_mixed = calculate_rv(df_mixed["StockPrice"]).mean()
    vol_finbert = calculate_rv(df_finbert["StockPrice"]).mean()

    print(f"Average Realized Volatility (Mixed): {vol_mixed:.4f}")
    print(f"Average Realized Volatility (FinBERT): {vol_finbert:.4f}")

    # Plot Stock Price comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df_mixed["StockPrice"], label="Mixed (VADER/FinBERT)", alpha=0.7)
    plt.plot(df_finbert["StockPrice"], label="Pure FinBERT", alpha=0.7)
    plt.title(
        f"Price Evolution: VADER vs FinBERT (Vol: {vol_mixed:.2f} vs {vol_finbert:.2f})"
    )
    plt.xlabel("Step")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig("figures/sentiment_comparison.png")
    print("Plot saved to figures/sentiment_comparison.png")


if __name__ == "__main__":
    test_hypothesis_1_divergence_spread()
    test_hypothesis_2_vol_smile()
    test_hypothesis_3_mm_hedging()
    test_sentiment_comparison()
