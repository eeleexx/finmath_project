import random

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector

from agents import MarketMaker, NoiseTrader, SophisticatedTrader
from utils import SentimentAnalyzer


class FinancialMarket(Model):
    def __init__(self, stock_symbol="AMD", num_noise=10, num_sophisticated=5, spread_pct=0.02):
        super().__init__()
        self.stock_symbol = stock_symbol
        self.num_noise = num_noise
        self.num_sophisticated = num_sophisticated
        self.spread_pct = spread_pct  # Bid-Ask Spread

        # Market Parameters
        self.stock_price = 100.0
        self.strike_price = 100.0  # At the money initially
        self.risk_free_rate = 0.02
        self.time_to_maturity = 1.0  # 1 Year
        self.dt = 1 / 252  # Daily steps
        self.realized_vol = 0.2

        # Load and Process Data
        self.news_data = self.load_data(stock_symbol)
        self.current_step_idx = 0
        self.current_news = None

        # Agents List (Manual Scheduling)
        self.market_agents = []

        # Market Maker
        self.market_maker = MarketMaker(self, spread_pct=self.spread_pct)
        self.market_agents.append(self.market_maker)

        for i in range(self.num_noise):
            a = NoiseTrader(self)
            self.market_agents.append(a)

        for i in range(self.num_sophisticated):
            a = SophisticatedTrader(self)
            self.market_agents.append(a)

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "StockPrice": "stock_price",
                "ImpliedVol_ITM": lambda m: m.market_maker.iv.get(int(m.stock_price * 0.9), 0.2),
                "ImpliedVol_ATM": lambda m: m.market_maker.iv.get(int(m.stock_price), 0.2),
                "ImpliedVol_OTM": lambda m: m.market_maker.iv.get(int(m.stock_price * 1.1), 0.2),
                "MMPortfolioValue": lambda m: m.market_maker.get_portfolio_value(),
                "TitleSentiment": lambda m: m.current_news["title_sentiment"]
                if m.current_news
                else 0,
                "ContentSentiment": lambda m: m.current_news["content_sentiment"]
                if m.current_news
                else 0,
                "Divergence": lambda m: abs(
                    m.current_news["title_sentiment"]
                    - m.current_news["content_sentiment"]
                )
                if m.current_news
                else 0,
                # Bid-Ask Spread Metrics
                "BidAskSpread_Call_ATM": lambda m: m.market_maker.get_current_spread()['call_spread'],
                "BidAskSpread_Put_ATM": lambda m: m.market_maker.get_current_spread()['put_spread'],
                "MMSpreadEarned": lambda m: m.market_maker.total_spread_earned,
                "MMCash": lambda m: m.market_maker.cash,
                "MMStockInventory": lambda m: m.market_maker.stock_inventory,
            }
        )

    def load_data(self, symbol):
        # Load news
        file_path = f"data/{symbol}_news.csv"
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found. Using empty.")
            return []

        # Analyze Sentiment
        analyzer = SentimentAnalyzer()
        # Handle possible missing columns or empty strings
        df["title"] = df["title"].fillna("")
        df["summary"] = df["summary"].fillna("")

        df["title_sentiment"] = df["title"].apply(analyzer.get_score)
        df["content_sentiment"] = df["summary"].apply(analyzer.get_score)

        return df.to_dict("records")

    def step(self):
        if self.current_step_idx >= len(self.news_data):
            self.running = False
            return

        # Get News
        self.current_news = self.news_data[self.current_step_idx]
        self.current_step_idx += 1

        # Update Stock Price (Geometric Brownian Motion)
        # Drift depends on Content Sentiment (The "Real" News)
        content_sent = self.current_news.get("content_sentiment", 0)
        # Base drift (risk-free) + Impact of Content
        drift = self.risk_free_rate + 0.5 * content_sent

        shock = np.random.normal(0, 1)
        self.stock_price *= np.exp(
            (drift - 0.5 * self.realized_vol**2) * self.dt
            + self.realized_vol * np.sqrt(self.dt) * shock
        )

        # Update Strikes dynamically (after price update)
        self.market_maker.update_strikes()

        # Agents Act (Random Order)
        random.shuffle(self.market_agents)
        for agent in self.market_agents:
            agent.step()

        # Collect Data
        self.datacollector.collect(self)
