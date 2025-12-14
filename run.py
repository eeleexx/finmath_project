import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model import FinancialMarket


def run_simulation():
    print("Starting Simulation for AMD...")
    # Initialize the model
    model = FinancialMarket(stock_symbol="AMD", num_noise=20, num_sophisticated=10)

    # Run for 252 steps (1 year) or until data ends
    for i in range(252):
        if not model.running:
            break
        model.step()

    print("Simulation finished.")

    # Get Data
    data = model.datacollector.get_model_vars_dataframe()

    if data.empty:
        print("No data collected!")
        return

    # Plot
    print("Plotting results...")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Stock Price
    sns.lineplot(data=data, x=data.index, y="StockPrice", ax=axes[0], color="blue")
    axes[0].set_title("Stock Price")
    axes[0].set_ylabel("Price ($)")

    # Implied Volatility
    # Implied Volatility
    sns.lineplot(data=data, x=data.index, y="ImpliedVol_ATM", ax=axes[1], color="red", label="ATM")
    sns.lineplot(data=data, x=data.index, y="ImpliedVol_ITM", ax=axes[1], color="green", linestyle="--", label="ITM")
    sns.lineplot(data=data, x=data.index, y="ImpliedVol_OTM", ax=axes[1], color="orange", linestyle="--", label="OTM")
    axes[1].set_title("Implied Volatility (IV)")
    axes[1].set_ylabel("IV")
    axes[1].legend()

    # Sentiment
    sns.lineplot(
        data=data,
        x=data.index,
        y="TitleSentiment",
        ax=axes[2],
        label="Title",
        alpha=0.6,
    )
    sns.lineplot(
        data=data,
        x=data.index,
        y="ContentSentiment",
        ax=axes[2],
        label="Content",
        alpha=0.6,
    )
    axes[2].set_title("News Sentiment")
    axes[2].set_ylabel("Sentiment Score")
    axes[2].legend()

    plt.tight_layout()
    output_path = "simulation_results.png"
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_simulation()
