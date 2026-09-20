"""
Виртуальный (paper trading) портфель.

ВАЖНО: этот модуль не содержит и не импортирует ничего, что может
разместить реальный ордер на бирже. Все сделки -- бухгалтерские записи
в памяти/на диске, привязанные к последней известной агенту цене.
Плечо/шорты не поддерживаются намеренно (упрощение для paper-режима);
для интеграции с реальным брокером/биржей этот класс не подходит и не
предназначен -- он только считает виртуальный P&L.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Trade:
    iteration: int
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    price: float
    fee: float
    cash_after: float
    reason: str


@dataclass
class PaperPortfolio:
    initial_cash: float
    cash: float = field(init=False)
    position_qty: float = 0.0
    position_avg_price: float = 0.0
    fee_pct: float = 0.0005
    trades: list = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def equity(self, current_price: float) -> float:
        return self.cash + self.position_qty * current_price

    def buy(self, iteration: int, symbol: str, price: float, cash_fraction: float, reason: str) -> Trade | None:
        spend = self.cash * cash_fraction
        if spend <= 0 or price <= 0:
            return None
        fee = spend * self.fee_pct
        qty = (spend - fee) / price
        if qty <= 0:
            return None

        new_total_qty = self.position_qty + qty
        self.position_avg_price = (
            (self.position_avg_price * self.position_qty + price * qty) / new_total_qty
            if new_total_qty > 0
            else 0.0
        )
        self.position_qty = new_total_qty
        self.cash -= spend

        trade = Trade(iteration, symbol, "buy", qty, price, fee, self.cash, reason)
        self.trades.append(trade)
        return trade

    def sell(self, iteration: int, symbol: str, price: float, position_fraction: float, reason: str) -> Trade | None:
        qty = self.position_qty * position_fraction
        if qty <= 0 or price <= 0:
            return None

        proceeds = qty * price
        fee = proceeds * self.fee_pct
        self.cash += proceeds - fee
        self.position_qty -= qty
        if self.position_qty <= 1e-12:
            self.position_qty = 0.0
            self.position_avg_price = 0.0

        trade = Trade(iteration, symbol, "sell", qty, price, fee, self.cash, reason)
        self.trades.append(trade)
        return trade

    def to_state_dict(self, current_price: float) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "position_qty": self.position_qty,
            "position_avg_price": self.position_avg_price,
            "current_price": current_price,
            "equity": self.equity(current_price),
            "n_trades": len(self.trades),
        }

    def save(self, path: str | Path, current_price: float) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.to_state_dict(current_price)
        state["trades"] = [asdict(t) for t in self.trades]
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False))
