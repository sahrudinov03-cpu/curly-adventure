"""
Генерация текста новостей об Ethereum для синтетических источников
(SimulatedEthNewsSource и офлайн-обучение в eth_training_data.py).

Категории событий подобраны по реальным типам новостей, двигающих ETH:
обновления протокола (Dencun/Pectra-подобные форки), регуляторные решения
(в т.ч. по спот-ETF на ETH), хаки DeFi-протоколов на Ethereum,
институциональные потоки, движения китов, партнёрства/L2-экосистема.
"""
import numpy as np

ETH_EVENT_TYPES = {
    # event_type -> базовая вероятность, что новость этого типа "бычья"
    "protocol_upgrade": 0.70,
    "hack": 0.05,
    "regulation_positive": 0.80,
    "regulation_negative": 0.10,
    "institutional_flow": 0.75,
    "whale_movement": 0.50,
    "partnership": 0.60,
}

POS_WORDS = [
    "рост", "апгрейд", "одобрение", "приток", "позитив", "прорыв",
    "поддержка", "внедрение", "бычий", "ралли", "рекорд", "стейкинг",
]
NEG_WORDS = [
    "взлом", "эксплойт", "падение", "запрет", "расследование", "отток",
    "паника", "негатив", "иск", "медвежий", "обвал", "уязвимость",
]
NEUTRAL_WORDS = [
    "ethereum", "eth", "газ", "валидаторы", "l2", "смарт-контракты",
    "биржа", "рынок", "трейдеры", "объём", "цена", "аналитики",
]


def make_news_text(rng: np.random.Generator, event_type: str, is_bullish: bool) -> str:
    n_pos = rng.integers(1, 4) if is_bullish else rng.integers(0, 2)
    n_neg = rng.integers(0, 2) if is_bullish else rng.integers(1, 4)
    words = (
        list(rng.choice(POS_WORDS, size=int(n_pos))) if n_pos else []
    ) + (
        list(rng.choice(NEG_WORDS, size=int(n_neg))) if n_neg else []
    ) + list(rng.choice(NEUTRAL_WORDS, size=int(rng.integers(2, 5))))
    rng.shuffle(words)
    return f"[{event_type}] " + " ".join(words)
