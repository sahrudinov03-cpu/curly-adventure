"""
Событийный анализ (event study): измерение аномальной доходности вокруг
новостных событий и проверка её статистической значимости.

Ожидаемая ("нормальная") доходность оценивается как средняя доходность
актива в окне до события -- на реальном рынке для альткоинов лучше
использовать регрессию на BTC (market model), но для одного актива
достаточно локальной средней.
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_abnormal_returns(
    price_df: pd.DataFrame,
    event_bar_idx: int,
    pre_window: int = 12,
    post_window: int = 24,
) -> pd.Series:
    """
    Возвращает кумулятивную аномальную доходность (CAR) по барам
    от -pre_window до +post_window относительно события.
    """
    closes = price_df["close"].values
    n = len(closes)
    lo = max(0, event_bar_idx - pre_window)
    hi = min(n - 1, event_bar_idx + post_window)

    window_returns = np.diff(np.log(closes[lo : hi + 1]))
    expected_return = np.mean(np.diff(np.log(closes[max(0, lo - pre_window) : lo + 1]))) if lo > 0 else 0.0

    abnormal = window_returns - expected_return
    car = np.cumsum(abnormal)
    offsets = np.arange(lo + 1, hi + 1) - event_bar_idx
    return pd.Series(car, index=offsets)


def event_study_by_type(
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    pre_window: int = 12,
    post_window: int = 24,
) -> pd.DataFrame:
    """
    Для каждого типа новости считает среднюю CAR на конец окна (+post_window)
    по всем историческим событиям этого типа и проверяет значимость через t-тест.
    """
    rows = []
    for etype, group in news_df.groupby("event_type"):
        final_cars = []
        for idx in group["bar_idx"]:
            car_series = compute_abnormal_returns(price_df, idx, pre_window, post_window)
            if post_window in car_series.index:
                final_cars.append(car_series.loc[post_window])

        if len(final_cars) < 2:
            continue

        final_cars = np.array(final_cars)
        t_stat, p_value = stats.ttest_1samp(final_cars, popmean=0.0)
        rows.append(
            {
                "event_type": etype,
                "n_events": len(final_cars),
                "mean_car": final_cars.mean(),
                "std_car": final_cars.std(ddof=1),
                "t_stat": t_stat,
                "p_value": p_value,
                "significant_5pct": p_value < 0.05,
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value")
    return result
