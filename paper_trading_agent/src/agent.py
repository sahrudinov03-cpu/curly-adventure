"""
Главный цикл агента бумажной торговли (paper trading).

Гарантии безопасности:
  - Никогда не импортирует и не вызывает ничего, что размещает реальный
    ордер (см. docstring в market_data.py и portfolio.py).
  - Стратегия видит только `price_history` -- буфер, который сам агент
    накопил по ходу опроса источника данных, т.е. исключительно "прошлые"
    для текущего момента наблюдения.
  - Виртуальный капитал существует только в PaperPortfolio (память/JSON),
    реальных денег и реальных биржевых аккаунтов агент не касается.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
from pathlib import Path

from .market_data import MarketDataSource, poll_loop
from .portfolio import PaperPortfolio
from .risk import RiskManager
from .strategy import Direction, SmaCrossoverStrategy

logger = logging.getLogger("paper_trading_agent")


class TradingAgent:
    def __init__(
        self,
        data_source: MarketDataSource,
        symbol: str,
        portfolio: PaperPortfolio,
        strategy: SmaCrossoverStrategy,
        risk_manager: RiskManager,
        state_dir: str | Path = "state",
        history_max_len: int = 500,
    ):
        self.data_source = data_source
        self.symbol = symbol
        self.portfolio = portfolio
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.history_max_len = history_max_len

        self.price_history: list[float] = []
        self._current_day = dt.date.today()
        self._log_path = self.state_dir / "trade_log.csv"
        self._init_log()

    def _init_log(self) -> None:
        if not self._log_path.exists():
            with self._log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "timestamp", "price", "signal", "confidence", "action", "cash", "position_qty", "equity"])

    def _log_row(self, iteration: int, price: float, signal_dir: str, confidence: float, action: str) -> None:
        equity = self.portfolio.equity(price)
        with self._log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [iteration, dt.datetime.now().isoformat(), price, signal_dir, f"{confidence:.4f}", action, f"{self.portfolio.cash:.2f}", f"{self.portfolio.position_qty:.6f}", f"{equity:.2f}"]
            )

    def _maybe_reset_day(self, equity: float) -> None:
        today = dt.date.today()
        if today != self._current_day:
            self._current_day = today
            self.risk_manager.reset_day(equity)

    def step(self, iteration: int, price: float) -> str:
        self.price_history.append(price)
        if len(self.price_history) > self.history_max_len:
            self.price_history = self.price_history[-self.history_max_len :]

        equity = self.portfolio.equity(price)
        self._maybe_reset_day(equity)
        halted = self.risk_manager.check_daily_halt(equity)

        action = "hold"

        # Управление уже открытой позицией: стоп-лосс / тейк-профит приоритетнее нового сигнала
        if self.portfolio.position_qty > 0:
            entry = self.portfolio.position_avg_price
            if self.risk_manager.should_stop_loss(entry, price):
                self.portfolio.sell(iteration, self.symbol, price, 1.0, "stop_loss")
                action = "sell_stop_loss"
            elif self.risk_manager.should_take_profit(entry, price):
                self.portfolio.sell(iteration, self.symbol, price, 1.0, "take_profit")
                action = "sell_take_profit"

        signal = self.strategy.generate_signal(self.price_history)

        if action == "hold" and not halted:
            if signal.direction == Direction.LONG and self.portfolio.position_qty == 0:
                fraction = self.risk_manager.entry_fraction(signal)
                if fraction > 0:
                    self.portfolio.buy(iteration, self.symbol, price, fraction, signal.reason)
                    action = "buy"
            elif signal.direction == Direction.FLAT and self.portfolio.position_qty > 0:
                # выход по сигналу, если не сработали стоп/тейк
                self.portfolio.sell(iteration, self.symbol, price, 1.0, "signal_flat")
                action = "sell_signal"

        self._log_row(iteration, price, signal.direction.value, signal.confidence, action)
        logger.info(
            "iter=%s price=%.4f signal=%s conf=%.2f action=%s equity=%.2f halted=%s",
            iteration, price, signal.direction.value, signal.confidence, action, equity, halted,
        )
        return action

    def run(self, interval_sec: float, max_iterations: int | None = None) -> None:
        for iteration, price in poll_loop(self.data_source, self.symbol, interval_sec, max_iterations):
            self.step(iteration, price)
            self.portfolio.save(self.state_dir / "portfolio.json", price)
