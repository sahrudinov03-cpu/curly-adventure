"""
Монте-Карло симуляция для оценки риска стратегии: распределение максимальной
просадки и вероятность разорения (risk of ruin) при заданном сайзинге позиций.
"""
import numpy as np


def simulate_equity_curves(
    trade_returns_pct: np.ndarray,
    n_simulations: int = 10_000,
    n_trades: int = 200,
    starting_capital: float = 1.0,
    ruin_threshold: float = 0.5,
    seed: int = 7,
) -> dict:
    """
    trade_returns_pct: исторические доходности сделок как доля от капитала
                        на сделку (например, +0.02 = +2%, -0.01 = -1%)
    ruin_threshold: доля от стартового капитала, при падении ниже которой
                    считаем, что произошло "разорение"
    """
    rng = np.random.default_rng(seed)
    max_drawdowns = np.empty(n_simulations)
    ruin_count = 0
    final_equities = np.empty(n_simulations)

    for i in range(n_simulations):
        sampled_returns = rng.choice(trade_returns_pct, size=n_trades, replace=True)
        equity = starting_capital * np.cumprod(1 + sampled_returns)
        running_max = np.maximum.accumulate(np.concatenate([[starting_capital], equity]))[1:]
        drawdown = (running_max - equity) / running_max
        max_drawdowns[i] = drawdown.max()
        final_equities[i] = equity[-1]
        if equity.min() <= starting_capital * ruin_threshold:
            ruin_count += 1

    return {
        "mean_max_drawdown": max_drawdowns.mean(),
        "p95_max_drawdown": np.percentile(max_drawdowns, 95),
        "p99_max_drawdown": np.percentile(max_drawdowns, 99),
        "risk_of_ruin": ruin_count / n_simulations,
        "median_final_equity": np.median(final_equities),
        "p5_final_equity": np.percentile(final_equities, 5),
    }
