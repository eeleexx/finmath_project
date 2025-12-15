from mesa.visualization import SolaraViz, make_plot_component

from model import FinancialMarket

# Define model parameters
model_params = {
    "stock_symbol": {
        "type": "Select",
        "value": "AMD",
        "values": ["AMD", "INTC", "JPM", "BAC", "PFE", "JNJ"],
        "label": "Stock Symbol",
    },
    "num_noise": {
        "type": "SliderInt",
        "value": 20,
        "min": 1,
        "max": 100,
        "label": "Noise Traders (Retail)",
    },
    "num_sophisticated": {
        "type": "SliderInt",
        "value": 5,
        "min": 1,
        "max": 50,
        "label": "Sophisticated Traders (Hedge Funds)",
    },
    "sentiment_key_title": {
        "type": "Select",
        "value": "vader_title",
        "values": ["vader_title", "finbert_title"],
        "label": "Headline AI Model",
    },
    "sentiment_key_content": {
        "type": "Select",
        "value": "finbert_content",
        "values": [
            "finbert_content",
            "vader_title",
        ],  # fall back to title if content missing
        "label": "Content AI Model",
    },
}

# Create plot components
# 1. Main Market Dynamics
price_plot = make_plot_component("StockPrice")

# 2. Volatility Smile Evolution (Fixed Keys)
iv_plot = make_plot_component(
    ["ImpliedVol_OTM_Put", "ImpliedVol_ATM", "ImpliedVol_OTM_Call"],
    post_process=lambda ax: ax.set_ylabel("Implied Volatility")
    or ax.legend(["Put OTM (0.9)", "ATM (1.0)", "Call OTM (1.1)"]),
)

# 3. Market Maker P&L (Gamification aspect: "Are they surviving?")
wealth_plot = make_plot_component("MMPortfolioValue")

# 4. Sentiment & Divergence
sentiment_plot = make_plot_component(["TitleSentiment", "ContentSentiment"])
divergence_plot = make_plot_component("Divergence")

# Initial model instance
model = FinancialMarket()

# Create the visualization page
page = SolaraViz(
    model,
    model_params=model_params,
    components=[price_plot, iv_plot, wealth_plot, sentiment_plot, divergence_plot],
    name="Financial Market Simulation: Information Asymmetry",
)
