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
    model = FinancialMarket(stock_symbol="AMD", num_noise=30, num_sophisticated=5)

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
    plt.xlabel("Sentiment Divergence |Title - Content|")
    plt.ylabel("Volatility Spread (IV_ATM - RV)")
    plt.savefig("hypothesis_1_result.png")
    print("Plot saved to hypothesis_1_result.png")


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

    # Scenario B: High Noise
    print("Running Scenario B: High Noise (50 agents)...")
    model_high = FinancialMarket(stock_symbol="AMD", num_noise=50, num_sophisticated=10)
    for _ in range(252):
        if not model_high.running:
            break
        model_high.step()
    df_high = model_high.datacollector.get_model_vars_dataframe()

    # Calculate Average IV Structure at end of simulation (or average over time)
    # Let's take the average over the simulation to see structural bias
    avg_iv_low = {
        "90 (Put OTM)": df_low["ImpliedVol_ITM"].mean(),
        "100 (ATM)": df_low["ImpliedVol_ATM"].mean(),
        "110 (Call OTM)": df_low["ImpliedVol_OTM"].mean(),
    }

    avg_iv_high = {
        "90 (Put OTM)": df_high["ImpliedVol_ITM"].mean(),
        "100 (ATM)": df_high["ImpliedVol_ATM"].mean(),
        "110 (Call OTM)": df_high["ImpliedVol_OTM"].mean(),
    }

    print("Average IV Structure (Low Noise):", avg_iv_low)
    print("Average IV Structure (High Noise):", avg_iv_high)

    # Plot comparison
    strikes = [90, 100, 110]
    ivs_low = [
        avg_iv_low["90 (Put OTM)"],
        avg_iv_low["100 (ATM)"],
        avg_iv_low["110 (Call OTM)"],
    ]
    ivs_high = [
        avg_iv_high["90 (Put OTM)"],
        avg_iv_high["100 (ATM)"],
        avg_iv_high["110 (Call OTM)"],
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        strikes,
        ivs_low,
        marker="o",
        label="Low Noise (Rational Market)",
        linestyle="--",
    )
    plt.plot(
        strikes,
        ivs_high,
        marker="o",
        label="High Noise (Speculative Market)",
        linewidth=2.5,
    )
    plt.title("Hypothesis 2: Volatility Smile Shape")
    plt.xlabel("Strike Price (Moneyness)")
    plt.ylabel("Implied Volatility")
    plt.xticks(strikes)
    plt.legend()
    plt.grid(True)
    plt.savefig("hypothesis_2_result.png")
    print("Plot saved to hypothesis_2_result.png")


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
    final_wealth_low = df_low["MMPortfolioValue"].iloc[-1]

    print(f"Final MM Wealth (High Noise): ${final_wealth_high:,.2f}")
    print(f"Final MM Wealth (Low Noise):  ${final_wealth_low:,.2f}")

    # Hypothesis: MM makes LESS money (or loses more) in High Noise scenario due to adverse selection/gamma risk
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
    plt.savefig("hypothesis_3_result.png")
    print("Plot saved to hypothesis_3_result.png")


if __name__ == "__main__":
    test_hypothesis_1_divergence_spread()
    test_hypothesis_2_vol_smile()
    test_hypothesis_3_mm_hedging()
