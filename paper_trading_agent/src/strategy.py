"""
Онлайновая (каузальная) стратегия: на каждом шаге видит только цены,
которые агент уже реально пронаблюдал в потоке (`price_history`),
никогда -- будущие значения. Это принципиально отличается от бэктеста,
где легко случайно подсмотреть в будущее; здесь буфер физически
пополняется по одному значению за раз в реальном времени.

Стратегия по умолчанию -- пересечение скользящих средних (SMA fast/slow)
с фильтром по силе отклонения. Простая и прозрачная специально: чтобы
было понятно, на основании чего агент принимает решение, а не потому что
это оптимальная стратегия для реальных денег.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from .bayesian import BayesianNewsModel
from .news_source import NewsItem
from .sentiment import LexiconSentimentScorer


class Direction(Enum):
    LONG = "long"
    FLAT = "flat"


@dataclass
class Signal:
    direction: Direction
    confidence: float  # [0, 1], используется риск-менеджером для сайзинга
    reason: str


class SmaCrossoverStrategy:
    def __init__(self, fast_window: int = 5, slow_window: int = 20, min_gap_pct: float = 0.001):
        if fast_window >= slow_window:
            raise ValueError("fast_window должен быть меньше slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.min_gap_pct = min_gap_pct

    def generate_signal(self, price_history: list[float]) -> Signal:
        if len(price_history) < self.slow_window:
            return Signal(Direction.FLAT, 0.0, "недостаточно истории для расчёта SMA")

        fast = sum(price_history[-self.fast_window :]) / self.fast_window
        slow = sum(price_history[-self.slow_window :]) / self.slow_window
        gap_pct = (fast - slow) / slow if slow != 0 else 0.0

        if gap_pct > self.min_gap_pct:
            confidence = min(abs(gap_pct) / (self.min_gap_pct * 5), 1.0)
            return Signal(Direction.LONG, confidence, f"SMA{self.fast_window} > SMA{self.slow_window} на {gap_pct:.4%}")

        return Signal(Direction.FLAT, 0.0, f"нет уверенного восходящего пересечения (gap={gap_pct:.4%})")


class NewsAwareStrategy:
    """
    Сигнал строится последовательным байесовским обновлением (Bayes' rule
    применяется к каждой новой новости, апостериорная вероятность
    предыдущего шага становится априорной для следующего -- см.
    news_strategy/README.md, раздел про байесовское обновление).

    Между новостями убеждение экспоненциально затухает к базовому prior
    модели (period half-life = decay_half_life_sec): влияние новости на
    рынок обычно отыгрывается за часы, а не остаётся навсегда, поэтому
    "старое" убеждение должно ослабевать, если новых подтверждений не было.

    price_history в generate_signal не используется -- по требованию решение
    строится только на новостном потоке; интерфейс сохранён для совместимости
    с TradingAgent.
    """

    def __init__(
        self,
        model: BayesianNewsModel,
        sentiment_scorer: LexiconSentimentScorer,
        decay_half_life_sec: float = 3600.0,
        entry_threshold: float = 0.58,
        exit_threshold: float = 0.50,
    ):
        if not entry_threshold > 0.5:
            raise ValueError("entry_threshold должен быть больше 0.5")
        self.model = model
        self.scorer = sentiment_scorer
        self.decay_half_life_sec = decay_half_life_sec
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        self.belief_up = model.prior_up
        self._last_update_ts = time.time()
        self.recent_news: list[dict] = []

    def _decay_towards_prior(self, now: float) -> None:
        dt = max(now - self._last_update_ts, 0.0)
        if dt <= 0 or self.decay_half_life_sec <= 0:
            return
        decay = 0.5 ** (dt / self.decay_half_life_sec)
        self.belief_up = self.model.prior_up + (self.belief_up - self.model.prior_up) * decay
        self._last_update_ts = now

    def on_news(self, item: NewsItem) -> None:
        self._decay_towards_prior(item.timestamp)

        bucket = f"{item.event_type}_{self.scorer.bucket(self.scorer.score(item.text))}"
        n_buckets = max(len(self.model._buckets), 1)
        lu = self.model.likelihood_up.get(bucket, 1.0 / n_buckets)
        ld = self.model.likelihood_down.get(bucket, 1.0 / n_buckets)

        prior = self.belief_up  # апостериорная вероятность предыдущего шага = prior для этого
        numerator = lu * prior
        denominator = numerator + ld * (1 - prior)
        self.belief_up = numerator / denominator if denominator > 0 else prior
        self._last_update_ts = item.timestamp

        self.recent_news.append(
            {"timestamp": item.timestamp, "event_type": item.event_type, "bucket": bucket, "belief_up_after": self.belief_up, "text": item.text}
        )
        self.recent_news = self.recent_news[-20:]

    def generate_signal(self, price_history: list[float]) -> Signal:
        self._decay_towards_prior(time.time())
        belief = self.belief_up

        if belief >= self.entry_threshold:
            confidence = min((belief - 0.5) / (self.entry_threshold - 0.5), 1.0)
            return Signal(Direction.LONG, confidence, f"байесовское P(рост)={belief:.3f} >= порога входа {self.entry_threshold}")
        if belief <= self.exit_threshold:
            return Signal(Direction.FLAT, 0.0, f"байесовское P(рост)={belief:.3f} <= порога выхода {self.exit_threshold}")
        return Signal(Direction.FLAT, 0.0, f"P(рост)={belief:.3f} в нейтральной зоне -- новых позиций не открываем")
