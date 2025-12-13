import numpy as np
from mesa import Agent

from utils import BlackScholes


class MarketParticipant(Agent):
    def __init__(self, model, initial_wealth=10000):
        super().__init__(model)
        self.wealth = initial_wealth
        self.portfolio = {"calls": 0, "puts": 0, "cash": initial_wealth}


class MarketMaker(Agent):
    def __init__(self, model, initial_iv=0.2):
        super().__init__(model)
        # IV is now a dictionary mapping Strike -> IV
        self.iv = {90: initial_iv, 100: initial_iv, 110: initial_iv}

        # Detailed inventory for hedging
        self.inventory_calls = {90: 0, 100: 0, 110: 0}
        self.inventory_puts = {90: 0, 100: 0, 110: 0}
        self.stock_inventory = 0

        self.cash = 0
        self.sensitivity = 0.05

    def step(self):
        # Delta Hedging Logic
        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate

        total_delta = 0.0

        # Calculate Delta from Calls
        for K, qty in self.inventory_calls.items():
            if qty != 0:
                iv = self.iv[K]
                d = BlackScholes.call_delta(S, K, T, r, iv)
                total_delta += qty * d

        # Calculate Delta from Puts
        for K, qty in self.inventory_puts.items():
            if qty != 0:
                iv = self.iv[K]
                d = BlackScholes.put_delta(S, K, T, r, iv)
                total_delta += qty * d

        # Add Stock Inventory (Delta = 1)
        total_delta += self.stock_inventory

        # Hedge if Delta is significant (e.g. > 0.5 share)
        if abs(total_delta) > 0.5:
            hedge_qty = -int(round(total_delta))
            if hedge_qty != 0:
                cost = hedge_qty * S
                self.stock_inventory += hedge_qty
                self.cash -= cost

    def get_portfolio_value(self):
        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate

        val = self.cash + self.stock_inventory * S

        for K, qty in self.inventory_calls.items():
            if qty != 0:
                price = BlackScholes.call_price(S, K, T, r, self.iv[K])
                val += qty * price

        for K, qty in self.inventory_puts.items():
            if qty != 0:
                price = BlackScholes.put_price(S, K, T, r, self.iv[K])
                val += qty * price

        return val

    def quote_options(self, S, K, T, r):
        # Get IV for this specific strike, default to ATM (100) if not found or nearest
        iv_k = self.iv.get(K, self.iv[100])

        c_price = BlackScholes.call_price(S, K, T, r, iv_k)
        p_price = BlackScholes.put_price(S, K, T, r, iv_k)
        return c_price, p_price

    def process_order(self, order_type, quantity, strike):
        # order_type: 'buy_call', 'sell_call', 'buy_put', 'sell_put'
        # quantity: number of contracts
        # strike: 90, 100, or 110

        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate

        if strike not in self.iv:
            strike = 100  # Default to ATM

        c_price, p_price = self.quote_options(S, strike, T, r)
        cost = 0

        # Adjust specific Strike IV and Update Detailed Inventory
        # Note: If MM "buys call" (from agent perspective, agent is SELLING to MM),
        # but the method name is process_order from AGENT's perspective.
        # Wait, let's check usage in NoiseTrader:
        # cost = self.model.market_maker.process_order("buy_call", 1, strike=110)
        # This implies the AGENT is buying. So MM is SELLING.
        # So 'buy_call' means Agent buys, MM sells.

        if order_type == "buy_call":
            # Agent buys call, MM sells call (Short Call)
            cost = c_price * quantity
            self.inventory_calls[strike] -= quantity
            self.cash += cost
            self.iv[strike] *= 1 + self.sensitivity * (quantity / 1000)

        elif order_type == "sell_call":
            # Agent sells call, MM buys call (Long Call)
            cost = -c_price * quantity
            self.inventory_calls[strike] += quantity
            self.cash -= abs(cost)
            self.iv[strike] *= 1 - self.sensitivity * (quantity / 1000)

        elif order_type == "buy_put":
            # Agent buys put, MM sells put (Short Put)
            cost = p_price * quantity
            self.inventory_puts[strike] -= quantity
            self.cash += cost
            self.iv[strike] *= 1 + self.sensitivity * (quantity / 1000)

        elif order_type == "sell_put":
            # Agent sells put, MM buys put (Long Put)
            cost = -p_price * quantity
            self.inventory_puts[strike] += quantity
            self.cash -= abs(cost)
            self.iv[strike] *= 1 - self.sensitivity * (quantity / 1000)

        # Clamp IV
        self.iv[strike] = max(0.05, min(self.iv[strike], 3.0))

        return abs(cost)


class NoiseTrader(MarketParticipant):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        news = self.model.current_news
        if news is None:
            return

        sentiment = news.get("title_sentiment", 0)
        action_threshold = 0.1

        # HYPOTHESIS 2: Noise Traders buy "Lottery Tickets" (OTM Options)
        # If Bullish -> Buy Call OTM (Strike 110)
        # If Bearish -> Buy Put OTM (Strike 90)

        if sentiment > action_threshold:
            # Buy Call OTM (110)
            cost = self.model.market_maker.process_order("buy_call", 1, strike=110)
            self.portfolio["calls"] += 1
            self.portfolio["cash"] -= cost
        elif sentiment < -action_threshold:
            # Buy Put OTM (90)
            cost = self.model.market_maker.process_order("buy_put", 1, strike=90)
            self.portfolio["puts"] += 1
            self.portfolio["cash"] -= cost


class SophisticatedTrader(MarketParticipant):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        news = self.model.current_news
        if news is None:
            return

        title_sent = news.get("title_sentiment", 0)
        content_sent = news.get("content_sentiment", 0)

        # Divergence
        divergence = abs(title_sent - content_sent)

        # HYPOTHESIS 1: Trade Divergence on ATM Volatility
        # If divergence is high, we expect IV to be overpriced relative to RV.
        # Sell Straddle ATM (Strike 100)
        if divergence > 0.3:
            # Short Call ATM
            cost_c = self.model.market_maker.process_order("sell_call", 1, strike=100)
            self.portfolio["calls"] -= 1
            self.portfolio["cash"] += cost_c

            # Short Put ATM
            cost_p = self.model.market_maker.process_order("sell_put", 1, strike=100)
            self.portfolio["puts"] -= 1
            self.portfolio["cash"] += cost_p
