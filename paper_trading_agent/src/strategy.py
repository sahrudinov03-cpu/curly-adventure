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

from dataclasses import dataclass
from enum import Enum


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
