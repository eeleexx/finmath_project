import numpy as np
from mesa import Agent

from utils import BlackScholes


class MarketParticipant(Agent):
    def __init__(self, model, initial_wealth=10000):
        super().__init__(model)
        self.wealth = initial_wealth
        self.portfolio = {"calls": 0, "puts": 0, "cash": initial_wealth}


class MarketMaker(Agent):
    def __init__(self, model, initial_iv=0.2, spread_pct=0.02):
        super().__init__(model)
        # IV is now a dictionary mapping Strike -> IV
        self.initial_iv = initial_iv
        self.spread_pct = spread_pct  # Bid-Ask Spread
        S = self.model.stock_price
        # IV is now a dictionary mapping Strike -> IV
        self.iv = {
            int(S * 0.9): initial_iv,
            int(S): initial_iv,
            int(S * 1.1): initial_iv
        }

        # Detailed inventory for hedging
        self.inventory_calls = {}
        self.inventory_puts = {}

        for K in self.iv:
            if K not in self.inventory_calls:
                self.inventory_calls[K] = 0
                self.inventory_puts[K] = 0
        self.stock_inventory = 0

        self.cash = 0
        self.sensitivity = 0.05
        
        # Statistics for spread analysis
        self.total_spread_earned = 0.0

        self.total_spread_earned = 0.0

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

    def update_strikes(self):
        S = self.model.stock_price
        new_iv = {
            int(S * 0.9): self.iv.get(int(S * 0.9), self.initial_iv),
            int(S):       self.iv.get(int(S), self.initial_iv),
            int(S * 1.1): self.iv.get(int(S * 1.1), self.initial_iv)
        }

        # Keep old strikes if we have inventory
        for K, qty in self.inventory_calls.items():
            if qty != 0 and K not in new_iv:
                new_iv[K] = self.iv.get(K, self.initial_iv)

        for K, qty in self.inventory_puts.items():
            if qty != 0 and K not in new_iv:
                new_iv[K] = self.iv.get(K, self.initial_iv)

        self.iv = new_iv

        for K in self.iv:
            if K not in self.inventory_calls:
                self.inventory_calls[K] = 0
                self.inventory_puts[K] = 0

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
        """
        Returns Bid and Ask prices for Call and Put.
        Returns: (call_bid, call_ask, put_bid, put_ask)
        """
        # Get IV for this specific strike, default to ATM (int(S)) if not found
        iv_k = self.iv.get(K, self.iv.get(int(S), 0.2))

        # Mid prices (theoretical Black-Scholes prices)
        c_mid = BlackScholes.call_price(S, K, T, r, iv_k)
        p_mid = BlackScholes.put_price(S, K, T, r, iv_k)
        
        # Bid-Ask Spread
        # Bid = price MM BUYS at (Agent sells) - lower than mid
        # Ask = price MM SELLS at (Agent buys) - higher than mid
        half_spread = self.spread_pct / 2
        
        call_bid = c_mid * (1 - half_spread)
        call_ask = c_mid * (1 + half_spread)
        put_bid = p_mid * (1 - half_spread)
        put_ask = p_mid * (1 + half_spread)
        
        return call_bid, call_ask, put_bid, put_ask

    def get_current_spread(self, strike=None):
        """
        Get current spread for a specific strike (for DataCollector).
        """
        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate
        
        if strike is None:
            strike = int(S)  # ATM
            
        call_bid, call_ask, put_bid, put_ask = self.quote_options(S, strike, T, r)
        
        return {
            'call_spread': call_ask - call_bid,
            'put_spread': put_ask - put_bid,
            'call_mid': (call_ask + call_bid) / 2,
            'put_mid': (put_ask + put_bid) / 2
        }

    def process_order(self, order_type, quantity, strike):
        # order_type: 'buy_call', 'sell_call', 'buy_put', 'sell_put'
        # quantity: number of contracts
        # strike: 90, 100, or 110

        S = self.model.stock_price
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate

        if strike not in self.iv:
            strike = int(S)  # Default to ATM
            # Ensure strike exists in IV map if it was missing (though update_strikes should handle this)
            if strike not in self.iv:
                 self.iv[strike] = self.initial_iv
                 self.inventory_calls[strike] = 0
                 self.inventory_puts[strike] = 0

        call_bid, call_ask, put_bid, put_ask = self.quote_options(S, strike, T, r)
        cost = 0
        spread_earned = 0

        # Adjust specific Strike IV and Update Detailed Inventory
        # Note: If MM "buys call" (from agent perspective, agent is SELLING to MM),
        # but the method name is process_order from AGENT's perspective.
        
        if order_type == "buy_call":
            # Agent buys call -> pays Ask (higher)
            # MM sells call (Short Call)
            cost = call_ask * quantity
            self.inventory_calls[strike] -= quantity
            self.cash += cost
            
            # MM earns spread: (Ask - Mid) * quantity
            call_mid = (call_ask + call_bid) / 2
            spread_earned = (call_ask - call_mid) * quantity
            
            self.iv[strike] *= 1 + self.sensitivity * (quantity / 1000)

        elif order_type == "sell_call":
            # Agent sells call -> receives Bid (lower)
            # MM buys call (Long Call)
            cost = -call_bid * quantity
            self.inventory_calls[strike] += quantity
            self.cash -= abs(cost)
            
            # MM earns spread: (Mid - Bid) * quantity
            call_mid = (call_ask + call_bid) / 2
            spread_earned = (call_mid - call_bid) * quantity
            
            self.iv[strike] *= 1 - self.sensitivity * (quantity / 1000)

        elif order_type == "buy_put":
            # Agent buys put -> pays Ask (higher)
            # MM sells put (Short Put)
            cost = put_ask * quantity
            self.inventory_puts[strike] -= quantity
            self.cash += cost
            
            # MM earns spread
            put_mid = (put_ask + put_bid) / 2
            spread_earned = (put_ask - put_mid) * quantity
            
            self.iv[strike] *= 1 + self.sensitivity * (quantity / 1000)

        elif order_type == "sell_put":
            # Agent sells put -> receives Bid (lower)
            # MM buys put (Long Put)
            cost = -put_bid * quantity
            self.inventory_puts[strike] += quantity
            self.cash -= abs(cost)
            
            # MM earns spread
            put_mid = (put_ask + put_bid) / 2
            spread_earned = (put_mid - put_bid) * quantity
            
            self.iv[strike] *= 1 - self.sensitivity * (quantity / 1000)

        # Clamp IV
        self.iv[strike] = max(0.05, min(self.iv[strike], 3.0))
        
        # Track spread earnings
        self.total_spread_earned += spread_earned

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
            # Buy Call OTM (S * 1.1)
            strike = int(self.model.stock_price * 1.1)
            cost = self.model.market_maker.process_order("buy_call", 1, strike=strike)
            self.portfolio["calls"] += 1
            self.portfolio["cash"] -= cost
        elif sentiment < -action_threshold:
            # Buy Put OTM (S * 0.9)
            strike = int(self.model.stock_price * 0.9)
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

        title_sent = news.get("title_sentiment", 0)
        content_sent = news.get("content_sentiment", 0)

        # Divergence
        divergence = abs(title_sent - content_sent)

        # HYPOTHESIS 1: Trade Divergence on ATM Volatility
        # If divergence is high, we expect IV to be overpriced relative to RV.
        # Sell Straddle ATM (Strike 100)
        if divergence > 0.3:
            # Продаёт Straddle на страйке 100 (ATM) 
            # Продажа Straddle означает продажу Call и Put на одном и том же страйке
            # ATM - мне безразницы исполнить опцион или купить акцию по ее текущей цене 
            
            # Short Call ATM
            strike = int(self.model.stock_price)
            cost_c = self.model.market_maker.process_order("sell_call", 1, strike=strike)
            self.portfolio["calls"] -= 1
            self.portfolio["cash"] += cost_c

            # Short Put ATM
            cost_p = self.model.market_maker.process_order("sell_put", 1, strike=strike)
            self.portfolio["puts"] -= 1
            self.portfolio["cash"] += cost_p
