"""
Простой событийный бэктест: на каждую новость с достаточно уверенным
байесовским сигналом открывается позиция (лонг/шорт), которая держится
фиксированный горизонт баров, затем закрывается по цене.

Это НЕ production-grade backtester (нет проскальзывания, комиссий за вход
по market/limit раздельно, частичного исполнения) -- цель модуля показать
методологию: сигнал -> сайзинг через Kelly -> PnL -> equity curve.
"""
import numpy as np
import pandas as pd

from .kelly import position_size


def run_backtest(
    price_df: pd.DataFrame,
    news_with_signal: pd.DataFrame,
    hold_bars: int = 24,
    confidence_threshold: float = 0.6,
    reward_risk_ratio: float = 1.5,
    starting_capital: float = 10_000.0,
    fee_pct: float = 0.0005,  # комиссия тейкера за сторону, типично 0.04-0.05% на фьючерсах
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> pd.DataFrame:
    closes = price_df["close"].values
    n = len(closes)
    capital = starting_capital
    trades = []

    for _, row in news_with_signal.iterrows():
        p_up = row["posterior_up"]
        idx = int(row["bar_idx"])
        exit_idx = idx + hold_bars
        if exit_idx >= n:
            continue

        direction = None
        p_win = None
        if p_up >= confidence_threshold:
            direction = "long"
            p_win = p_up
        elif p_up <= 1 - confidence_threshold:
            direction = "short"
            p_win = 1 - p_up
        else:
            continue  # сигнал недостаточно уверенный -- не торгуем

        size = position_size(p_win, reward_risk_ratio, capital, kelly_multiplier, max_fraction)
        if size <= 0:
            continue

        entry_price = closes[idx]
        exit_price = closes[exit_idx]
        raw_return = (exit_price / entry_price - 1) if direction == "long" else (entry_price / exit_price - 1)
        net_return = raw_return - 2 * fee_pct  # комиссия на вход и выход

        pnl = size * net_return
        capital += pnl

        trades.append(
            {
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "direction": direction,
                "p_win_est": p_win,
                "size": size,
                "return_pct": net_return,
                "pnl": pnl,
                "capital_after": capital,
            }
        )

    return pd.DataFrame(trades)


def summarize_backtest(trades_df: pd.DataFrame, starting_capital: float = 10_000.0) -> dict:
    if trades_df.empty:
        return {"n_trades": 0}

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    final_capital = trades_df["capital_after"].iloc[-1]

    equity = np.concatenate([[starting_capital], trades_df["capital_after"].values])
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max

    return {
        "n_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df),
        "avg_win_pct": wins["return_pct"].mean() if len(wins) else 0.0,
        "avg_loss_pct": losses["return_pct"].mean() if len(losses) else 0.0,
        "total_return_pct": (final_capital / starting_capital - 1) * 100,
        "max_drawdown_pct": drawdown.max() * 100,
        "final_capital": final_capital,
    }
