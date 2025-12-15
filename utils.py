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


class FinBERTAnalyzer:
    """
    Finance-specific sentiment analyzer using ProsusAI/finbert.
    Returns sentiment score in range [-1, 1] for consistency with VADER.
    """
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def get_score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                probs = self.torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            positive = probs[0][0].item()
            negative = probs[0][1].item()
            
            # Convert to [-1, 1] scale: positive - negative
            return positive - negative
        except Exception:
            return 0.0

