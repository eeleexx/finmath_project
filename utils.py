import re

import numpy as np
from scipy.stats import norm


class BlackScholes:
    @staticmethod
    def d1(S, K, T, r, sigma):
        if sigma <= 0 or T <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        if sigma <= 0 or T <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def call_delta(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1)

    @staticmethod
    def put_delta(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_score(self, text: str) -> float:
        if not isinstance(text, str):
            return 0.0

        scores = self.analyzer.polarity_scores(text)
        return scores["compound"]
