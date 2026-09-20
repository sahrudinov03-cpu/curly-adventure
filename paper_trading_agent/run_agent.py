"""
Запуск агента бумажной торговли (paper trading -- виртуальные деньги, без
реальных ордеров на бирже).

Примеры:
    # Демо на симулированных данных (работает без доступа в интернет)
    python run_agent.py --data-source simulated --symbol DEMOUSDT \
        --interval 1 --max-iterations 60

    # Реальный live-фид цен с публичного API Binance (нужен доступ в
    # интернет; ключи биржи НЕ нужны и НЕ используются -- это read-only
    # публичный эндпоинт котировок, размещение ордеров не поддерживается)
    python run_agent.py --data-source live --symbol BTCUSDT --interval 15
"""
import argparse
import logging

from src.agent import TradingAgent
from src.market_data import build_data_source
from src.portfolio import PaperPortfolio
from src.risk import RiskLimits, RiskManager
from src.strategy import SmaCrossoverStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper trading agent (виртуальные деньги, без реальных ордеров)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-source", choices=["live", "simulated"], default="simulated")
    p.add_argument("--interval", type=float, default=15.0, help="секунд между опросами цены")
    p.add_argument("--max-iterations", type=int, default=None, help="ограничение числа шагов (без него -- бесконечный цикл)")
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--fee-pct", type=float, default=0.0005)
    p.add_argument("--fast-window", type=int, default=5)
    p.add_argument("--slow-window", type=int, default=20)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    p.add_argument("--stop-loss-pct", type=float, default=0.03)
    p.add_argument("--take-profit-pct", type=float, default=0.06)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.05)
    p.add_argument("--state-dir", default="state")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    data_source = build_data_source(args.data_source)
    portfolio = PaperPortfolio(initial_cash=args.initial_cash, fee_pct=args.fee_pct)
    strategy = SmaCrossoverStrategy(fast_window=args.fast_window, slow_window=args.slow_window)
    risk_manager = RiskManager(
        RiskLimits(
            risk_per_trade_pct=args.risk_per_trade_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            daily_loss_limit_pct=args.daily_loss_limit_pct,
        )
    )

    agent = TradingAgent(
        data_source=data_source,
        symbol=args.symbol,
        portfolio=portfolio,
        strategy=strategy,
        risk_manager=risk_manager,
        state_dir=args.state_dir,
    )

    print(f"Старт paper-trading агента: {args.symbol}, источник={args.data_source}, начальный капитал={args.initial_cash}")
    print(f"Лог сделок: {args.state_dir}/trade_log.csv, состояние портфеля: {args.state_dir}/portfolio.json")
    print("Это СИМУЛЯЦИЯ на виртуальные деньги. Реальные ордера не размещаются.")

    try:
        agent.run(interval_sec=args.interval, max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")


if __name__ == "__main__":
    main()
