import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "thyroidDF.csv"

# Тест 1. Проверка маппинга
def test_gender_mapping():
    df = pd.read_csv(DATA_PATH, nrows=10)
    df["sex"] = df["sex"].map({"F": 1, "M": 0})

    unique_values = df["sex"].unique().tolist()

    assert set(unique_values).issubset({0, 1}), "Ошибка: в колонке пола остались неперекодированные значения"

# Тест 2. Проверка корректности формата предсказаний модели
def test_predictions_format():
    X = np.random.randn(10, 4)
    y = np.random.choice([0, 1, 2], size=10)

    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = model.predict(X[:3])

    assert isinstance(preds, np.ndarray), "Ошибка: предсказания модели должны быть массивом"
    assert len(preds) == 3, "Ошибка: количество предсказаний не совпадает с размером тестовой выборки"
    assert set(preds).issubset({0, 1, 2}), "Ошибка: модель выдала несуществующий класс"
