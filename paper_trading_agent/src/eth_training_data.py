"""
Офлайн-обучение байесовской модели на исторических данных ETH перед
запуском live-цикла -- ровно тот же принцип, что в news_strategy/:
event study отбирает статистически значимые типы новостей, затем
BayesianNewsModel.fit оценивает правдоподобия на train-периоде.

Здесь история синтетическая (см. news_text_gen.py и ограничение сети в
README) -- при переходе на реальные данные замените `_generate_eth_history`
на загрузку исторических новостей CryptoPanic + цен ETH и постройте те же
DataFrame с колонками timestamp/bar_idx/event_type/text (новости) и
timestamp/close (цены).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bayesian import BayesianNewsModel
from .event_study import event_study_by_type
from .news_text_gen import ETH_EVENT_TYPES, make_news_text
from .sentiment import LexiconSentimentScorer

# (p_bullish, средний шок, шум) по типу события -- откалибровано так, чтобы
# относительный порядок эффектов совпадал со здравым смыслом: хаки сильно
# негативны, регуляторный позитив и институциональные потоки заметно
# позитивны, движения китов -- почти чистый шум.
SHOCK_PROFILE = {
    "protocol_upgrade": (0.70, 0.015, 0.012),
    "hack": (0.05, 0.045, 0.02),
    "regulation_positive": (0.80, 0.025, 0.015),
    "regulation_negative": (0.10, 0.03, 0.018),
    "institutional_flow": (0.75, 0.02, 0.014),
    "whale_movement": (0.50, 0.012, 0.02),
    "partnership": (0.60, 0.010, 0.012),
}


def _generate_eth_history(n_bars: int = 4000, n_events: int = 260, bar_minutes: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n_bars, freq=f"{bar_minutes}min")
    base_returns = rng.normal(0.0, 0.003, n_bars)

    event_bar_idx = rng.choice(np.arange(20, n_bars - 20), size=n_events, replace=False)
    event_bar_idx.sort()

    events = []
    for idx in event_bar_idx:
        etype = rng.choice(list(SHOCK_PROFILE.keys()))
        p_bull, mean_shock, noise = SHOCK_PROFILE[etype]
        is_bullish = rng.random() < p_bull
        shock = abs(rng.normal(mean_shock, noise))
        shock = shock if is_bullish else -shock
        base_returns[idx] += shock

        events.append(
            {
                "timestamp": timestamps[idx],
                "bar_idx": idx,
                "event_type": etype,
                "text": make_news_text(rng, etype, is_bullish),
            }
        )

    prices = 100 * np.exp(np.cumsum(base_returns))
    price_df = pd.DataFrame({"timestamp": timestamps, "close": prices})
    news_df = pd.DataFrame(events)
    return price_df, news_df


def train_eth_model(horizon_bars: int = 24, seed: int = 11) -> dict:
    """
    Возвращает: {"model": BayesianNewsModel, "scorer": LexiconSentimentScorer,
    "event_study": pd.DataFrame} -- обученную модель и отчёт о значимости
    типов новостей (печатается перед запуском live-агента, чтобы решение
    агента было прослеживаемо к статистике, а не к "чёрному ящику").
    """
    price_df, news_df = _generate_eth_history(seed=seed)

    scorer = LexiconSentimentScorer()
    news_df["sentiment_bucket"] = news_df["text"].apply(lambda t: scorer.bucket(scorer.score(t)))
    news_df["feature"] = news_df["event_type"] + "_" + news_df["sentiment_bucket"]

    closes = price_df["close"].values
    outcomes = []
    for idx in news_df["bar_idx"]:
        exit_idx = idx + horizon_bars
        outcomes.append(closes[exit_idx] > closes[idx] if exit_idx < len(closes) else np.nan)
    news_df["outcome_up"] = outcomes
    news_df = news_df.dropna(subset=["outcome_up"]).reset_index(drop=True)

    event_report = event_study_by_type(price_df, news_df)

    model = BayesianNewsModel().fit(news_df["feature"], news_df["outcome_up"])

    return {"model": model, "scorer": scorer, "event_study": event_report}
