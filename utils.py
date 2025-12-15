import numpy as np
from scipy.stats import norm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# ---------- VADER ----------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_score(self, text: str) -> float:
        if not isinstance(text, str) or text.strip() == "":
            return 0.0
        return self.analyzer.polarity_scores(text)["compound"]


# ---------- FinBERT ----------
class FinBertSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def get_score(self, text: str) -> float:
        if not isinstance(text, str) or text.strip() == "":
            return 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()

        # labels: [negative, neutral, positive]
        score = probs[2] - probs[0]
        return float(score)