"""
Критерий Келли для сайзинга позиций на основе вероятностной оценки успеха.

f* = p - (1 - p) / b

где p -- оценённая вероятность успеха сделки (из байесовской модели),
b -- отношение прибыль/убыток (reward/risk ratio).

На практике используется дробный Келли (fraction < 1), потому что оценка p
всегда содержит ошибку, а полный Келли крайне чувствителен к её переоценке.
"""


def kelly_fraction(p_win: float, reward_risk_ratio: float) -> float:
    if reward_risk_ratio <= 0:
        raise ValueError("reward_risk_ratio должен быть положительным")
    f = p_win - (1 - p_win) / reward_risk_ratio
    return max(f, 0.0)  # отрицательный Келли = не входить в сделку


def position_size(
    p_win: float,
    reward_risk_ratio: float,
    capital: float,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """
    kelly_multiplier: доля от полного Келли (0.25 = "quarter Kelly", рекомендуется)
    max_fraction: жёсткий потолок доли капитала на сделку вне зависимости от Келли
    """
    f = kelly_fraction(p_win, reward_risk_ratio) * kelly_multiplier
    f = min(f, max_fraction)
    return capital * f
