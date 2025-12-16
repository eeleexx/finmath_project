import random

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector

from agents import MarketMaker, NoiseTrader, SophisticatedTrader


class FinancialMarket(Model):
    def __init__(
        self,
        stock_symbol="AMD",
        num_noise=10,
        num_sophisticated=5,
        sentiment_key_title="vader_title",
        sentiment_key_content="finbert_content",
    ):
        super().__init__()
        self.stock_symbol = stock_symbol
        self.num_noise = num_noise
        self.num_sophisticated = num_sophisticated
        self.sentiment_key_title = sentiment_key_title
        self.sentiment_key_content = sentiment_key_content

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
        self.market_maker = MarketMaker(self)
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
                "ImpliedVol_OTM_Put": lambda m: m.market_maker.iv_buckets.get(0.9, 0.2),
                "ImpliedVol_ATM": lambda m: m.market_maker.iv_buckets.get(1.0, 0.2),
                "ImpliedVol_OTM_Call": lambda m: m.market_maker.iv_buckets.get(
                    1.1, 0.2
                ),
                "MMPortfolioValue": lambda m: m.market_maker.get_portfolio_value(),
                "TitleSentiment": lambda m: m.current_news.get(m.sentiment_key_title, 0)
                if m.current_news
                else 0,
                "ContentSentiment": lambda m: m.current_news.get(
                    m.sentiment_key_content, 0
                )
                if m.current_news
                else 0,
                "Divergence": lambda m: abs(
                    m.current_news.get(m.sentiment_key_title, 0)
                    - m.current_news.get(m.sentiment_key_content, 0)
                )
                if m.current_news
                else 0,
            }
        )

    def load_data(self, symbol):
        # Paths to sentiment files
        base_path = "sentiments"
        vader_title_path = f"{base_path}/{symbol}_vader_title.csv"
        finbert_title_path = f"{base_path}/{symbol}_finbert_title.csv"
        finbert_content_path = f"{base_path}/{symbol}_finbert_content.csv"

        try:
            # Load VADER Title (Base DataFrame)
            df = pd.read_csv(vader_title_path)
            df = df.rename(columns={"sentiment": "vader_title"})

            # Load FinBERT Title
            if pd.io.common.file_exists(finbert_title_path):
                df_ft = pd.read_csv(finbert_title_path)
                df["finbert_title"] = df_ft["sentiment"]
            else:
                df["finbert_title"] = 0.0

            # Load FinBERT Content
            if pd.io.common.file_exists(finbert_content_path):
                df_fc = pd.read_csv(finbert_content_path)
                df["finbert_content"] = df_fc["sentiment"]
            else:
                df["finbert_content"] = 0.0

        except FileNotFoundError:
            print(
                f"Sentiment files for {symbol} not found in {base_path}. Using empty."
            )
            return []

        # Ensure no NaNs in sentiments
        df["vader_title"] = df["vader_title"].fillna(0.0)
        df["finbert_title"] = df["finbert_title"].fillna(0.0)
        df["finbert_content"] = df["finbert_content"].fillna(0.0)

        # Keep other useful columns
        if "title" not in df.columns:
            df["title"] = ""
        if "summary" not in df.columns:
            df["summary"] = ""

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
        content_sent = self.current_news.get(self.sentiment_key_content, 0)
        # Base drift (risk-free) + Impact of Content
        drift = self.risk_free_rate + 0.5 * content_sent

        shock = np.random.normal(0, 1)
        self.stock_price *= np.exp(
            (drift - 0.5 * self.realized_vol**2) * self.dt
            + self.realized_vol * np.sqrt(self.dt) * shock
        )

        # Agents Act (Random Order)
        random.shuffle(self.market_agents)
        for agent in self.market_agents:
            agent.step()

        # Collect Data
        self.datacollector.collect(self)
