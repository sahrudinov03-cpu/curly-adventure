"""
Оценка сентимента новостного текста.

Реализован простой лексиконный скорер (прозрачный, без внешних зависимостей
и скачивания моделей) -- достаточен для демонстрации методологии на
синтетических данных.

Для продакшена замените `LexiconSentimentScorer` на обёртку над
transformers-моделью, например:

    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", model="ElKulako/cryptobert")

    class CryptoBertScorer:
        def score(self, text: str) -> float:
            result = classifier(text)[0]
            sign = 1.0 if result["label"] == "Bullish" else -1.0
            return sign * result["score"]

Интерфейс (`score(text) -> float в диапазоне [-1, 1]`) остаётся тем же,
поэтому остальной пайплайн менять не нужно.
"""
from dataclasses import dataclass, field

POSITIVE_WORDS = {
    "рост", "партнёрство", "листинг", "одобрение", "позитив", "прорыв",
    "поддержка", "внедрение", "бычий", "ралли", "рекорд",
}
NEGATIVE_WORDS = {
    "взлом", "падение", "запрет", "расследование", "хак", "паника",
    "негатив", "иск", "медвежий", "обвал", "мошенничество",
}


@dataclass
class LexiconSentimentScorer:
    positive_words: set = field(default_factory=lambda: set(POSITIVE_WORDS))
    negative_words: set = field(default_factory=lambda: set(NEGATIVE_WORDS))

    def score(self, text: str) -> float:
        tokens = text.lower().replace("[", " ").replace("]", " ").split()
        pos = sum(1 for t in tokens if t in self.positive_words)
        neg = sum(1 for t in tokens if t in self.negative_words)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def bucket(self, score: float, n_buckets: int = 3) -> str:
        """Дискретизация скора для байесовской модели (нужны частотные таблицы)."""
        if n_buckets == 3:
            if score > 0.2:
                return "positive"
            if score < -0.2:
                return "negative"
            return "neutral"
        raise NotImplementedError("Поддерживается только n_buckets=3")
