"""
Риск-менеджер: превращает сигнал стратегии в конкретный размер сделки и
следит за стоп-лоссом/тейк-профитом и дневным лимитом убытков -- те же
принципы, что в руководстве по риск-менеджменту (1-2% риска на сделку,
обязательный стоп-лосс, дневной лимит).
"""
from __future__ import annotations

from dataclasses import dataclass

from .strategy import Direction, Signal


@dataclass
class RiskLimits:
    max_position_fraction: float = 0.5  # не более 50% капитала в одной позиции
    risk_per_trade_pct: float = 0.02  # доля от capital-риска на сделку, масштабирует размер входа через confidence
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    daily_loss_limit_pct: float = 0.05


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._day_start_equity: float | None = None
        self._halted = False

    def check_daily_halt(self, current_equity: float) -> bool:
        if self._day_start_equity is None:
            self._day_start_equity = current_equity
            return self._halted
        drawdown = (self._day_start_equity - current_equity) / self._day_start_equity
        if drawdown >= self.limits.daily_loss_limit_pct:
            self._halted = True
        return self._halted

    def reset_day(self, current_equity: float) -> None:
        self._day_start_equity = current_equity
        self._halted = False

    def entry_fraction(self, signal: Signal) -> float:
        """Доля СВОБОДНОГО КЭША, которую можно направить на вход, исходя из уверенности сигнала."""
        if signal.direction != Direction.LONG or self._halted:
            return 0.0
        fraction = self.limits.risk_per_trade_pct * (1 + 4 * signal.confidence)  # confidence масштабирует риск в [1x, 5x] базового
        return min(fraction, self.limits.max_position_fraction)

    def should_stop_loss(self, entry_price: float, current_price: float) -> bool:
        if entry_price <= 0:
            return False
        return (current_price / entry_price - 1) <= -self.limits.stop_loss_pct

    def should_take_profit(self, entry_price: float, current_price: float) -> bool:
        if entry_price <= 0:
            return False
        return (current_price / entry_price - 1) >= self.limits.take_profit_pct
