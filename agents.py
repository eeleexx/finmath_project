from collections import deque

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
        # IV buckets based on Moneyness (K/S)
        # 0.9: OTM Put / ITM Call
        # 1.0: ATM
        # 1.1: OTM Call / ITM Put
        self.iv_buckets = {0.9: initial_iv, 1.0: initial_iv, 1.1: initial_iv}

        # Detailed inventory for hedging: Strike -> Quantity
        self.inventory_calls = {}
        self.inventory_puts = {}
        self.stock_inventory = 0

        self.cash = 0
        self.sensitivity = 0.05
        self.spread = 0.02  # 2% Bid-Ask Spread

    def get_iv(self, moneyness):
        # Find nearest bucket
        available = list(self.iv_buckets.keys())
        nearest = min(available, key=lambda x: abs(x - moneyness))
        return self.iv_buckets[nearest]

    def update_iv(self, moneyness, factor):
        available = list(self.iv_buckets.keys())
        nearest = min(available, key=lambda x: abs(x - moneyness))
        self.iv_buckets[nearest] *= factor
        # Clamp IV
        self.iv_buckets[nearest] = max(0.05, min(self.iv_buckets[nearest], 3.0))

    def step(self):
        # Delta Hedging Logic
        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate

        total_delta = 0.0

        # Calculate Delta from Calls
        for K, qty in self.inventory_calls.items():
            if qty != 0:
                moneyness = K / S if S > 0 else 1.0
                iv = self.get_iv(moneyness)
                d = BlackScholes.call_delta(S, K, T, r, iv)
                total_delta += qty * d

        # Calculate Delta from Puts
        for K, qty in self.inventory_puts.items():
            if qty != 0:
                moneyness = K / S if S > 0 else 1.0
                iv = self.get_iv(moneyness)
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
                moneyness = K / S if S > 0 else 1.0
                price = BlackScholes.call_price(S, K, T, r, self.get_iv(moneyness))
                val += qty * price

        for K, qty in self.inventory_puts.items():
            if qty != 0:
                moneyness = K / S if S > 0 else 1.0
                price = BlackScholes.put_price(S, K, T, r, self.get_iv(moneyness))
                val += qty * price

        return val

    def quote_options(self, S, K, T, r):
        moneyness = K / S if S > 0 else 1.0
        iv = self.get_iv(moneyness)

        mid_c = BlackScholes.call_price(S, K, T, r, iv)
        mid_p = BlackScholes.put_price(S, K, T, r, iv)

        # Apply Spread
        # Buy at Ask, Sell at Bid
        c_ask = mid_c * (1 + self.spread / 2)
        c_bid = mid_c * (1 - self.spread / 2)

        p_ask = mid_p * (1 + self.spread / 2)
        p_bid = mid_p * (1 - self.spread / 2)

        return (c_bid, c_ask), (p_bid, p_ask)

    def process_order(self, order_type, quantity, strike):
        # order_type: 'buy_call', 'sell_call', 'buy_put', 'sell_put'
        # quantity: number of contracts
        # strike: Dynamic strike price

        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate

        (c_bid, c_ask), (p_bid, p_ask) = self.quote_options(S, strike, T, r)
        cost = 0
        moneyness = strike / S if S > 0 else 1.0

        if order_type == "buy_call":
            # Agent buys call @ Ask, MM sells
            cost = c_ask * quantity
            self.inventory_calls[strike] = (
                self.inventory_calls.get(strike, 0) - quantity
            )
            self.cash += cost
            self.update_iv(moneyness, 1 + self.sensitivity * (quantity / 1000))

        elif order_type == "sell_call":
            # Agent sells call @ Bid, MM buys
            cost = -c_bid * quantity  # Negative cost for agent (income)
            self.inventory_calls[strike] = (
                self.inventory_calls.get(strike, 0) + quantity
            )
            self.cash -= abs(cost)
            self.update_iv(moneyness, 1 - self.sensitivity * (quantity / 1000))

        elif order_type == "buy_put":
            # Agent buys put @ Ask
            cost = p_ask * quantity
            self.inventory_puts[strike] = self.inventory_puts.get(strike, 0) - quantity
            self.cash += cost
            self.update_iv(moneyness, 1 + self.sensitivity * (quantity / 1000))

        elif order_type == "sell_put":
            # Agent sells put @ Bid
            cost = -p_bid * quantity
            self.inventory_puts[strike] = self.inventory_puts.get(strike, 0) + quantity
            self.cash -= abs(cost)
            self.update_iv(moneyness, 1 - self.sensitivity * (quantity / 1000))

        return abs(cost)


class NoiseTrader(MarketParticipant):
    def __init__(self, model):
        super().__init__(model)
        self.sentiment_history = deque(maxlen=20)

    def step(self):
        news = self.model.current_news
        if news is None:
            return

        # Use configured sentiment key
        sentiment_key = getattr(self.model, "sentiment_key_title", "title_sentiment")
        sentiment = news.get(sentiment_key, 0)

        self.sentiment_history.append(abs(sentiment))

        # Dynamic Threshold: Mean of recent absolute sentiment + base buffer
        if len(self.sentiment_history) > 0:
            avg_vol = sum(self.sentiment_history) / len(self.sentiment_history)
            action_threshold = max(0.05, avg_vol * 1.2)  # Dynamic
        else:
            action_threshold = 0.1

        S = self.model.stock_price

        if sentiment > action_threshold:
            # Bullish: Buy Call OTM (Moneyness 1.1)
            strike = round(S * 1.1, 1)
            cost = self.model.market_maker.process_order("buy_call", 1, strike=strike)
            self.portfolio["calls"] += 1
            self.portfolio["cash"] -= cost

        elif sentiment < -action_threshold:
            # Bearish: Buy Put OTM (Moneyness 0.9)
            strike = round(S * 0.9, 1)
            cost = self.model.market_maker.process_order("buy_put", 1, strike=strike)
            self.portfolio["puts"] += 1
            self.portfolio["cash"] -= cost


class SophisticatedTrader(MarketParticipant):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        news = self.model.current_news
        if news is None:
            return

        key_title = getattr(self.model, "sentiment_key_title", "title_sentiment")
        key_content = getattr(self.model, "sentiment_key_content", "content_sentiment")

        title_sent = news.get(key_title, 0)
        content_sent = news.get(key_content, 0)

        # Divergence
        divergence = abs(title_sent - content_sent)

        S = self.model.stock_price

        # Trade Divergence on ATM Volatility
        if divergence > 0.3:
            strike = round(S, 1)  # ATM

            # Short Call ATM
            cost_c = self.model.market_maker.process_order(
                "sell_call", 1, strike=strike
            )
            self.portfolio["calls"] -= 1
            self.portfolio["cash"] += (
                cost_c  # Received premium (positive cash flow effectively, but cost is returned as abs in process_order, wait. process_order returns abs(cost). logic in agents needs to handle direction.)
            )

            # The previous implementation:
            # cost_c = process_order(...) -> returns abs value
            # self.portfolio["cash"] += cost_c
            # Correct.

            # Short Put ATM
            cost_p = self.model.market_maker.process_order("sell_put", 1, strike=strike)
            self.portfolio["puts"] -= 1
            self.portfolio["cash"] += cost_p
