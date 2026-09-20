"""
Байесовское обновление вероятности роста цены на основе категории сентимента.

P(рост | сентимент) = P(сентимент | рост) * P(рост) / P(сентимент)

Правдоподобия P(сентимент | рост) и P(сентимент | падение) оцениваются
эмпирически на обучающей выборке (частотные таблицы), затем применяются
к новым новостям на тестовой выборке.
"""
from collections import defaultdict

import pandas as pd


class BayesianNewsModel:
    def __init__(self, prior_up: float = 0.5, laplace_smoothing: float = 1.0):
        self.prior_up = prior_up
        self.alpha = laplace_smoothing
        self.likelihood_up: dict[str, float] = {}
        self.likelihood_down: dict[str, float] = {}
        self._buckets: list[str] = []

    def fit(self, sentiment_buckets: pd.Series, outcome_up: pd.Series) -> "BayesianNewsModel":
        """
        sentiment_buckets: категория сентимента ('positive'/'neutral'/'negative')
        outcome_up: bool -- была ли доходность после новости положительной
        """
        self._buckets = sorted(sentiment_buckets.unique())
        counts_up = defaultdict(int)
        counts_down = defaultdict(int)
        for bucket, up in zip(sentiment_buckets, outcome_up):
            if up:
                counts_up[bucket] += 1
            else:
                counts_down[bucket] += 1

        n_up = sum(counts_up.values())
        n_down = sum(counts_down.values())
        k = len(self._buckets)

        for b in self._buckets:
            self.likelihood_up[b] = (counts_up[b] + self.alpha) / (n_up + self.alpha * k)
            self.likelihood_down[b] = (counts_down[b] + self.alpha) / (n_down + self.alpha * k)

        # эмпирический prior из обучающей выборки, если не задан явно
        total = n_up + n_down
        if total > 0:
            self.prior_up = n_up / total
        return self

    def posterior_up(self, bucket: str) -> float:
        lu = self.likelihood_up.get(bucket, 1.0 / max(len(self._buckets), 1))
        ld = self.likelihood_down.get(bucket, 1.0 / max(len(self._buckets), 1))
        numerator = lu * self.prior_up
        denominator = numerator + ld * (1 - self.prior_up)
        if denominator == 0:
            return self.prior_up
        return numerator / denominator

    def predict(self, sentiment_buckets: pd.Series) -> pd.Series:
        return sentiment_buckets.map(self.posterior_up)
