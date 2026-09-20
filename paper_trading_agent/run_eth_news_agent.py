"""
Paper-trading агент на ETH, решения которого строятся ИСКЛЮЧИТЕЛЬНО на
новостном потоке по Ethereum через последовательное байесовское обновление
(см. src/strategy.py:NewsAwareStrategy).

Перед запуском live-цикла модель офлайн "обучается" на исторических данных
(здесь -- синтетических, см. src/eth_training_data.py) и печатает отчёт
event study: какие типы ETH-новостей вообще статистически значимо двигают
цену -- чтобы было видно, на чём основано решение агента, а не только
итоговую цифру P(рост).

Примеры:
    # Демо: симулированные цена + новости (без доступа в интернет)
    python run_eth_news_agent.py --data-source simulated --news-source simulated \
        --interval 1 --max-iterations 300

    # Live: реальные котировки Binance + реальная лента CryptoPanic по ETH
    # (нужен доступ в интернет и бесплатный CRYPTOPANIC_TOKEN;
    # токен даёт только чтение новостей, не биржевой аккаунт)
    CRYPTOPANIC_TOKEN=... python run_eth_news_agent.py \
        --data-source live --news-source live --interval 30
"""
import argparse
import logging
import os

from src.agent import TradingAgent
from src.eth_training_data import train_eth_model
from src.market_data import build_data_source
from src.news_source import CryptoPanicNewsSource, SimulatedEthNewsSource
from src.portfolio import PaperPortfolio
from src.risk import RiskLimits, RiskManager
from src.strategy import NewsAwareStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH paper-trading агент на новостном байесовском сигнале")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--data-source", choices=["live", "simulated"], default="simulated")
    p.add_argument("--news-source", choices=["live", "simulated"], default="simulated")
    p.add_argument("--cryptopanic-token", default=os.environ.get("CRYPTOPANIC_TOKEN"))
    p.add_argument("--interval", type=float, default=15.0, help="секунд между опросами цены/новостей")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--fee-pct", type=float, default=0.0005)
    p.add_argument("--entry-threshold", type=float, default=0.58, help="P(рост) >= порога -> вход в лонг")
    p.add_argument("--exit-threshold", type=float, default=0.50, help="P(рост) <= порога -> выход из лонга")
    p.add_argument("--decay-half-life-sec", type=float, default=3600.0, help="за сколько секунд влияние новости затухает вдвое")
    p.add_argument("--news-rate-per-hour", type=float, default=3.0, help="только для simulated news-source")
    p.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    p.add_argument("--stop-loss-pct", type=float, default=0.03)
    p.add_argument("--take-profit-pct", type=float, default=0.06)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.05)
    p.add_argument("--state-dir", default="state_eth")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("Офлайн-обучение байесовской модели на исторических ETH-новостях")
    print("=" * 70)
    trained = train_eth_model()
    model, scorer, event_report = trained["model"], trained["scorer"], trained["event_study"]

    print(event_report.to_string(index=False))
    print(f"\nPrior P(рост) по всей истории: {model.prior_up:.3f}")

    if args.news_source == "live":
        if not args.cryptopanic_token:
            raise SystemExit("--news-source live требует --cryptopanic-token или переменную окружения CRYPTOPANIC_TOKEN")
        news_source = CryptoPanicNewsSource(api_token=args.cryptopanic_token, currency="ETH")
    else:
        news_source = SimulatedEthNewsSource(news_rate_per_hour=args.news_rate_per_hour)

    data_source = build_data_source(args.data_source)
    portfolio = PaperPortfolio(initial_cash=args.initial_cash, fee_pct=args.fee_pct)
    strategy = NewsAwareStrategy(
        model=model,
        sentiment_scorer=scorer,
        decay_half_life_sec=args.decay_half_life_sec,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
    )
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
        news_source=news_source,
    )

    print()
    print(f"Старт ETH news-агента: {args.symbol}, цена={args.data_source}, новости={args.news_source}, капитал={args.initial_cash}")
    print(f"Лог сделок: {args.state_dir}/trade_log.csv, лог новостей: {args.state_dir}/news_log.csv")
    print("Это СИМУЛЯЦИЯ на виртуальные деньги. Реальные ордера не размещаются.")

    try:
        agent.run(interval_sec=args.interval, max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")


if __name__ == "__main__":
    main()
