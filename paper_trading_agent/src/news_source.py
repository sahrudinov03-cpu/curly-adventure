"""
Источники новостей по ETH.

Как и market_data.py: источник отдаёт только те новости, которые уже
реально произошли на момент опроса -- ни один из них не может вернуть
новость "из будущего". CryptoPanicNewsSource использует только read-only
эндпоинт (auth_token нужен исключительно для чтения ленты, доступа к
ордерам или счёту он не даёт и не может дать).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import requests

from .news_text_gen import ETH_EVENT_TYPES, make_news_text


@dataclass
class NewsItem:
    timestamp: float  # unix ts, когда агент реально это увидел
    event_type: str
    text: str


class NewsDataSource(ABC):
    @abstractmethod
    def poll_new(self) -> list[NewsItem]:
        """Возвращает новости, появившиеся с прошлого вызова poll_new()."""


class CryptoPanicNewsSource(NewsDataSource):
    """
    Публичная лента CryptoPanic (https://cryptopanic.com/developers/api/),
    отфильтрованная по валюте ETH. auth_token берётся бесплатно на сайте
    CryptoPanic -- это ключ на ЧТЕНИЕ новостей, он не связан с биржевым
    аккаунтом и не может размещать ордера.

    Категория события определяется грубой эвристикой по заголовку -- для
    продакшена стоит заменить на классификатор (см. news_strategy/README.md,
    раздел про NLP-скоринг) либо использовать поле `kind`/`votes` API.
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_token: str, currency: str = "ETH", timeout_sec: float = 5.0):
        self.api_token = api_token
        self.currency = currency
        self.timeout_sec = timeout_sec
        self._seen_ids: set[str] = set()

    def poll_new(self) -> list[NewsItem]:
        resp = requests.get(
            self.BASE_URL,
            params={"auth_token": self.api_token, "currencies": self.currency, "public": "true"},
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()

        items = []
        for post in data.get("results", []):
            pid = str(post.get("id"))
            if not pid or pid in self._seen_ids:
                continue
            self._seen_ids.add(pid)
            items.append(
                NewsItem(
                    timestamp=time.time(),
                    event_type=_infer_event_type(post.get("title", "")),
                    text=post.get("title", ""),
                )
            )
        return items


def _infer_event_type(title: str) -> str:
    t = title.lower()
    if "hack" in t or "exploit" in t or "vulnerab" in t:
        return "hack"
    if "sec" in t or "regulat" in t or "law" in t or "ban" in t:
        return "regulation_negative" if ("ban" in t or "sue" in t or "charge" in t) else "regulation_positive"
    if "upgrade" in t or "fork" in t or "eip" in t:
        return "protocol_upgrade"
    if "etf" in t or "institution" in t or "inflow" in t or "outflow" in t:
        return "institutional_flow"
    if "whale" in t:
        return "whale_movement"
    if "partner" in t:
        return "partnership"
    return "partnership"  # нейтральная категория по умолчанию для нераспознанного заголовка


class SimulatedEthNewsSource(NewsDataSource):
    """
    Симулирует появление ETH-новостей как пуассоновский процесс с
    интенсивностью `news_rate_per_hour` -- на каждом poll_new() вероятность
    получить новость за истекшее реальное время dt равна 1 - exp(-λ·dt).
    Используется там, где нет сетевого доступа к CryptoPanic (см. README).
    """

    def __init__(self, news_rate_per_hour: float = 3.0, seed: int = 123):
        self._rng = np.random.default_rng(seed)
        self._lambda_per_sec = news_rate_per_hour / 3600.0
        self._last_poll = time.time()

    def poll_new(self) -> list[NewsItem]:
        now = time.time()
        dt = max(now - self._last_poll, 0.0)
        self._last_poll = now

        p_event = 1 - np.exp(-self._lambda_per_sec * dt) if dt > 0 else 0.0
        if self._rng.random() >= p_event:
            return []

        event_type = self._rng.choice(list(ETH_EVENT_TYPES.keys()))
        is_bullish = self._rng.random() < ETH_EVENT_TYPES[event_type]
        text = make_news_text(self._rng, event_type, is_bullish)
        return [NewsItem(timestamp=now, event_type=event_type, text=text)]
