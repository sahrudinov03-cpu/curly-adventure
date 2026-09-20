"""
Генерирует синтетические данные (цены + новости) для демонстрации пайплайна.

В реальном использовании этот скрипт заменяется на:
- цены: выгрузка с биржи (Binance/Bybit REST API, интервал 5m/15m)
- новости: CryptoPanic API / Twitter API / RSS-парсеры (см. README.md)

Синтетика намеренно содержит встроенную (но зашумлённую) зависимость между
типом новости и последующим движением цены, чтобы пайплайн мог статистически
её обнаружить -- это проверка того, что методы вообще способны находить
сигнал, когда он есть.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Тип события -> (вероятность что событие "бычье", средний шок в %, шум)
EVENT_TYPES = {
    "listing": (0.75, 0.025, 0.015),
    "hack": (0.05, 0.04, 0.02),
    "regulation_positive": (0.80, 0.02, 0.015),
    "regulation_negative": (0.10, 0.03, 0.02),
    "partnership": (0.60, 0.012, 0.012),
    "whale_movement": (0.50, 0.015, 0.02),
}

POS_WORDS = ["рост", "партнёрство", "листинг", "одобрение", "позитив", "прорыв", "поддержка", "внедрение"]
NEG_WORDS = ["взлом", "падение", "запрет", "расследование", "хак", "паника", "негатив", "иск"]
NEUTRAL_WORDS = ["криптовалюта", "биржа", "актив", "рынок", "трейдеры", "объём", "цена", "аналитики"]


def make_news_text(event_type: str, is_bullish: bool) -> str:
    n_pos = RNG.integers(1, 4) if is_bullish else RNG.integers(0, 2)
    n_neg = RNG.integers(0, 2) if is_bullish else RNG.integers(1, 4)
    words = (
        list(RNG.choice(POS_WORDS, size=n_pos))
        + list(RNG.choice(NEG_WORDS, size=n_neg))
        + list(RNG.choice(NEUTRAL_WORDS, size=RNG.integers(2, 5)))
    )
    RNG.shuffle(words)
    return f"[{event_type}] " + " ".join(words)


def generate(n_bars: int = 4000, n_events: int = 300, bar_minutes: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start, periods=n_bars, freq=f"{bar_minutes}min")

    # базовое случайное блуждание (log-returns), близкое к нулевому дрифту
    base_returns = rng.normal(loc=0.0, scale=0.003, size=n_bars)

    event_bar_idx = rng.choice(np.arange(20, n_bars - 20), size=n_events, replace=False)
    event_bar_idx.sort()

    events = []
    for idx in event_bar_idx:
        etype = rng.choice(list(EVENT_TYPES.keys()))
        p_bull, mean_shock, noise = EVENT_TYPES[etype]
        is_bullish = rng.random() < p_bull
        shock = rng.normal(mean_shock, noise)
        shock = abs(shock) if is_bullish else -abs(shock)
        base_returns[idx] += shock
        events.append(
            {
                "timestamp": timestamps[idx],
                "bar_idx": idx,
                "event_type": etype,
                "text": make_news_text(etype, is_bullish),
                "true_is_bullish": is_bullish,  # скрытая правда, для валидации методологии
            }
        )

    prices = 100 * np.exp(np.cumsum(base_returns))
    price_df = pd.DataFrame({"timestamp": timestamps, "close": prices})
    news_df = pd.DataFrame(events)
    return price_df, news_df


if __name__ == "__main__":
    price_df, news_df = generate()
    price_df.to_csv("data/prices_sample.csv", index=False)
    news_df.to_csv("data/news_sample.csv", index=False)
    print(f"Сгенерировано {len(price_df)} баров цены и {len(news_df)} новостных событий")
    print("Сохранено в data/prices_sample.csv и data/news_sample.csv")
