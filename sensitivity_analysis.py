import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from model import FinancialMarket


def calculate_rv(price_series, window=20):
    log_returns = np.log(price_series / price_series.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def run_sensitivity_analysis():
    print("Starting Sensitivity Analysis on Agent Populations...")

    noise_levels = [10, 50, 100]
    soph_levels = [5, 20, 50]

    results = []

    for n_noise in noise_levels:
        for n_soph in soph_levels:
            print(f"Running scenario: Noise={n_noise}, Sophisticated={n_soph}...")

            model = FinancialMarket(
                stock_symbol="AMD",
                num_noise=n_noise,
                num_sophisticated=n_soph,
                sentiment_key_title="vader_title",
                sentiment_key_content="finbert_content",
            )

            # Run simulation
            for _ in range(252):
                if not model.running:
                    break
                model.step()

            df = model.datacollector.get_model_vars_dataframe()

            # 1. Metric: Volatility Smile Strength (OTM Call IV - ATM IV)
            # We take the mean over the simulation
            smile_strength = (df["ImpliedVol_OTM_Call"] - df["ImpliedVol_ATM"]).mean()

            # 2. Metric: Market Maker Final Wealth
            mm_wealth = df["MMPortfolioValue"].iloc[-1]

            # 3. Metric: H1 Correlation (Divergence vs Spread)
            df["RV"] = calculate_rv(df["StockPrice"])
            df["Spread"] = df["ImpliedVol_ATM"] - df["RV"]
            clean_df = df.dropna(subset=["Spread", "Divergence"])

            if len(clean_df) > 10:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    clean_df["Divergence"], clean_df["Spread"]
                )
                spread_corr = r_value
                spread_pval = p_value
            else:
                spread_corr = 0
                spread_pval = 1.0

            results.append(
                {
                    "Noise Traders": n_noise,
                    "Sophisticated Traders": n_soph,
                    "Smile Strength": smile_strength,
                    "MM Wealth": mm_wealth,
                    "Spread Correlation": spread_corr,
                    "Spread P-Value": spread_pval,
                }
            )

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    res_df.to_csv("results/results_sensitivity.csv", index=False)
    print("Sensitivity results saved to results/results_sensitivity.csv")

    # --- PLOTTING HEATMAPS ---
    sns.set_theme(style="whitegrid")

    # 1. Smile Strength Heatmap
    pivot_smile = res_df.pivot(
        index="Noise Traders", columns="Sophisticated Traders", values="Smile Strength"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_smile, annot=True, cmap="YlOrRd", fmt=".4f")
    plt.title("Impact on Volatility Smile (OTM - ATM IV)")
    plt.savefig("sensitivity_smile.png")
    plt.close()

    # 2. MM Wealth Heatmap
    pivot_wealth = res_df.pivot(
        index="Noise Traders", columns="Sophisticated Traders", values="MM Wealth"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_wealth, annot=True, cmap="RdYlGn", fmt=",.0f")
    plt.title("Impact on Market Maker Wealth ($)")
    plt.savefig("sensitivity_wealth.png")
    plt.close()

    # 3. Spread Correlation Heatmap
    pivot_corr = res_df.pivot(
        index="Noise Traders",
        columns="Sophisticated Traders",
        values="Spread Correlation",
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_corr, annot=True, cmap="coolwarm", fmt=".4f", center=0)
    plt.title("Impact on Divergence-Spread Correlation")
    plt.savefig("sensitivity_correlation.png")
    plt.close()

    print(
        "Plots saved: sensitivity_smile.png, sensitivity_wealth.png, sensitivity_correlation.png"
    )


if __name__ == "__main__":
    run_sensitivity_analysis()
