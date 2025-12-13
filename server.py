import solara
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
        "value": 10,
        "min": 1,
        "max": 50,
        "label": "Noise Traders",
    },
    "num_sophisticated": {
        "type": "SliderInt",
        "value": 5,
        "min": 1,
        "max": 50,
        "label": "Sophisticated Traders",
    },
}

# Create plot components
price_plot = make_plot_component("StockPrice")
# Plot all three strikes to see the Smile/Skew evolution
iv_plot = make_plot_component(["ImpliedVol_90", "ImpliedVol_100", "ImpliedVol_110"])
sentiment_plot = make_plot_component(["TitleSentiment", "ContentSentiment"])

# Initial model instance
model = FinancialMarket()

# Create the visualization page
page = SolaraViz(
    model,
    model_params=model_params,
    components=[price_plot, iv_plot, sentiment_plot],
    name="Financial Market Simulation",
)
