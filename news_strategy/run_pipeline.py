"""
Полный демонстрационный прогон пайплайна:

  новости -> сентимент -> байесовская вероятность -> событийный анализ
  -> сайзинг (Kelly) -> бэктест -> риск-оценка (Монте-Карло)

Запуск:
    pip install -r requirements.txt
    python generate_sample_data.py
    python run_pipeline.py

Все данные в data/ -- синтетические (см. generate_sample_data.py). Для
реального использования замените источники данных согласно README.md,
интерфейсы модулей (score/fit/predict/run_backtest) менять не нужно.
"""
import numpy as np
import pandas as pd

from src.backtest import run_backtest, summarize_backtest
from src.bayesian import BayesianNewsModel
from src.event_study import event_study_by_type
from src.monte_carlo import simulate_equity_curves
from src.sentiment import LexiconSentimentScorer


def main():
    price_df = pd.read_csv("data/prices_sample.csv", parse_dates=["timestamp"])
    news_df = pd.read_csv("data/news_sample.csv", parse_dates=["timestamp"])

    # 1. Сентимент-скоринг
    scorer = LexiconSentimentScorer()
    news_df["sentiment_score"] = news_df["text"].apply(scorer.score)
    news_df["sentiment_bucket"] = news_df["sentiment_score"].apply(scorer.bucket)

    # Признак для байесовской модели: тип события + сентимент. Один сентимент
    # (см. раздел 2 ниже) даёт слабую сепарацию -- реальный эффект в новостях
    # чаще привязан к категории события (хак/листинг/регуляция), а не только
    # к тональности текста, поэтому комбинируем оба признака.
    news_df["feature"] = news_df["event_type"] + "_" + news_df["sentiment_bucket"]

    # 2. Разметка исхода: выросла ли цена через 24 бара после новости
    horizon = 24
    closes = price_df["close"].values
    outcomes = []
    for idx in news_df["bar_idx"]:
        exit_idx = idx + horizon
        if exit_idx < len(closes):
            outcomes.append(closes[exit_idx] > closes[idx])
        else:
            outcomes.append(np.nan)
    news_df["outcome_up"] = outcomes
    news_df = news_df.dropna(subset=["outcome_up"]).reset_index(drop=True)

    # 3. Train/test split по времени (walk-forward, без заглядывания в будущее)
    split_idx = int(len(news_df) * 0.6)
    train, test = news_df.iloc[:split_idx], news_df.iloc[split_idx:]

    print("=" * 70)
    print("1. СОБЫТИЙНЫЙ АНАЛИЗ (event study) -- какие типы новостей значимы")
    print("=" * 70)
    event_stats = event_study_by_type(price_df, news_df)
    print(event_stats.to_string(index=False))

    print()
    print("=" * 70)
    print("2. БАЙЕСОВСКАЯ МОДЕЛЬ -- обучение на train, оценка на test")
    print("=" * 70)
    model = BayesianNewsModel().fit(train["feature"], train["outcome_up"])
    print(f"Prior P(рост) на train: {model.prior_up:.3f}")
    posteriors_train = {b: model.posterior_up(b) for b in sorted(model.likelihood_up)}
    for bucket, p in sorted(posteriors_train.items(), key=lambda kv: abs(kv[1] - model.prior_up), reverse=True)[:8]:
        print(f"  P(рост | {bucket}) posterior = {p:.3f}  (n признак редок -> сглажено к prior)")

    test = test.copy()
    test["posterior_up"] = model.predict(test["feature"])

    # Калибровка на test: действительно ли высокая posterior -> выше реальный win rate
    calibration = (
        test.assign(bucket=pd.cut(test["posterior_up"], bins=[0, 0.4, 0.6, 1.0]))
        .groupby("bucket", observed=True)["outcome_up"]
        .agg(["mean", "count"])
    )
    print("\nКалибровка модели на test-выборке (насколько posterior соответствует факту):")
    print(calibration.rename(columns={"mean": "фактическая_доля_роста", "count": "n"}).to_string())

    print()
    print("=" * 70)
    print("3. БЭКТЕСТ на test-выборке")
    print("=" * 70)
    trades = run_backtest(price_df, test, hold_bars=horizon, confidence_threshold=0.55)
    summary = summarize_backtest(trades)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if len(trades) >= 5:
        print()
        print("=" * 70)
        print("4. МОНТЕ-КАРЛО -- риск разорения при данном распределении сделок")
        print("=" * 70)
        # доля капитала на сделку (size / стартовый капитал) x реализованная доходность
        capital_fraction_returns = (trades["size"] / 10_000.0) * trades["return_pct"]
        mc = simulate_equity_curves(capital_fraction_returns.values, n_trades=200)
        for k, v in mc.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nНедостаточно сделок в бэктесте для устойчивой Монте-Карло оценки.")


if __name__ == "__main__":
    main()
