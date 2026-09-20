"""
Источники рыночных данных для агента.

Все источники отдают только ту цену, которая реально "уже произошла" на
момент вызова -- ни один из них не может заглянуть в будущее. Это важно:
агент принимает решения исключительно на основе данных, которые он сам
успел пронаблюдать в потоке (онлайн/каузальная логика), а не на основе
заранее известного исхода.

BinancePublicDataSource ходит в публичный REST-эндпоинт биржи (без ключей,
без прав на торговлю -- физически не может разместить реальный ордер).
Использовать для настоящего live-режима на машине с доступом в интернет.

SimulatedDataSource генерирует цену локально (random walk) -- для разработки
и тестов в окружениях без доступа к бирже, а также для прогона в CI.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import requests


class MarketDataSource(ABC):
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Возвращает последнюю известную цену. Никогда не данные "из будущего"."""


class BinancePublicDataSource(MarketDataSource):
    """
    Публичный REST-эндпоинт Binance (ticker/price) -- не требует API-ключа,
    не предоставляет доступа к размещению ордеров. Только чтение цены.
    """

    BASE_URL = "https://api.binance.com/api/v3/ticker/price"

    def __init__(self, timeout_sec: float = 5.0):
        self.timeout_sec = timeout_sec

    def get_latest_price(self, symbol: str) -> float:
        resp = requests.get(self.BASE_URL, params={"symbol": symbol}, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        return float(data["price"])


class SimulatedDataSource(MarketDataSource):
    """
    Локальный генератор цены (geometric random walk) для тестирования логики
    агента без сетевого доступа к бирже. Не использовать для реальных решений
    о капитале -- только для проверки работоспособности пайплайна.
    """

    def __init__(self, start_price: float = 100.0, volatility: float = 0.002, seed: int = 0):
        import random

        self._rng = random.Random(seed)
        self._price = start_price
        self._volatility = volatility

    def get_latest_price(self, symbol: str) -> float:
        shock = self._rng.gauss(0, self._volatility)
        self._price *= (1 + shock)
        self._price = max(self._price, 0.01)
        return self._price


def build_data_source(kind: str, **kwargs) -> MarketDataSource:
    if kind == "live":
        return BinancePublicDataSource(**kwargs)
    if kind == "simulated":
        return SimulatedDataSource(**kwargs)
    raise ValueError(f"Неизвестный источник данных: {kind}")


def poll_loop(source: MarketDataSource, symbol: str, interval_sec: float, max_iterations: int | None = None):
    """Генератор: отдаёт (iteration, price) с заданным интервалом опроса."""
    i = 0
    while max_iterations is None or i < max_iterations:
        price = source.get_latest_price(symbol)
        yield i, price
        i += 1
        if max_iterations is None or i < max_iterations:
            time.sleep(interval_sec)
