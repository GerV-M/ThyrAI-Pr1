import logging
import sys
import warnings

import dvc.api
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Логирование
log = logging.getLogger("ThyrAI")
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

warnings.filterwarnings("ignore")

# Чтение файлов, обработка датасетов
from pathlib import Path

PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"
params = dvc.api.params_show(repo=str(PARAMS_PATH.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data"
OUTPUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "results"

# Уровень ERROR
try:
    df1 = pd.read_csv(DATA_DIR / "thyroidDF.csv")
    df2 = pd.read_csv(DATA_DIR / "hypothyroid.csv")
    log.info("Датасеты успешно загружены.")
except FileNotFoundError as e:
    log.error(f"Датасеты не обнаружены. {e}")
    sys.exit(1)

df2.replace('?', np.nan, inplace=True)

df1 = df1.drop(columns=['referral_source', 'query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG', 'patient_id'])
df2 = df2.drop(columns=['referral source', 'query on thyroxine', 'query hypothyroid', 'query hyperthyroid', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured', 'TBG'])

df2 = df2.rename(columns={'binaryClass':'target', 'on thyroxine' : 'on_thyroxine', 'on antithyroid medication' : 'on_antithyroid_meds', 'thyroid surgery' : 'thyroid_surgery', 'I131 treatment' : 'I131_treatment'})

target_map = {'-': 0, 'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 2, 'F': 2, 'G': 2, 'H': 2, 'P': 2}
df1 = df1[df1['target'].isin(target_map.keys())].copy()
df1['target'] = df1['target'].map(target_map)

df2 = df2[df2['target'].isin(target_map.keys())].copy()
df2['target'] = df2['target'].map(target_map)

numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
categorical_cols = [col for col in df1.columns if col not in numeric_cols + ['target']]

for col in numeric_cols:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
    df2[col] = pd.to_numeric(df2[col], errors='coerce')

# Объединение датасетов, очистка от выбросов в возрасте и от повторных значений
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['age'] < 120.0]
df = df.drop_duplicates(subset=['age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])

# Подготовка данных, стандартизация
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = params['preparation']['test_size'],
                                                    random_state = params['preparation']['random_seed'],
                                                    stratify=y)

# Заполнение пустых значений
mode = X_train[categorical_cols].mode().iloc[0]
X_train[categorical_cols] = X_train[categorical_cols].fillna(mode)
X_test[categorical_cols] = X_test[categorical_cols].fillna(mode)

imputer = KNNImputer(n_neighbors=params['preparation']['knn_neighbors'], weights='distance')

X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

# Бинаризация
binary_cols = X_train.select_dtypes(include='object').columns

for col in binary_cols:
    if col != 'target':
        X_train[col] = X_train[col].map({'t':1, 'f':0, 'F':1, 'M':0})
        X_test[col] = X_test[col].map({'t':1, 'f':0, 'F':1, 'M':0})

# Скэйлер
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Гендер в числа для матриц корреляции
df['sex'] = df['sex'].map({'F':1, 'M':0})


import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

results = []

# Случайный лес
log.info("Запуск обучения модели Случайного леса...")

# Уровень WARNING
if params['random_forest']['class_weight'] is None:
    log.warning("Параметр class_weight равен None. На исходных датасетах высока вероятность переобучения модели.")

start_time = time.perf_counter()

rf_model = RandomForestClassifier(max_depth=params['random_forest']['max_depth'],
                                  max_features=params['random_forest']['max_features'],
                                  n_estimators=params['random_forest']['n_estimators'],
                                  random_state=params['preparation']['random_seed'],
                                  class_weight=params['random_forest']['class_weight'],
                                  n_jobs=-1)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

end_time = time.perf_counter()

# Уровень INFO
log.info(f"Время обучения: {end_time - start_time:.4f} секунд.")

log.info(f"Accuracy: {accuracy_score(y_test, rf_pred)}")
log.info(f"Precision: {precision_score(y_test, rf_pred, average='macro')}")
log.info(f"Recall: {recall_score(y_test, rf_pred, average='macro')}")
log.info(f"F1-score: {f1_score(y_test, rf_pred, average='macro')}")

y_proba = rf_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

log.info(f"ROC-AUC: {roc_auc}")

print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

results.append({
    "Model": "Случайный лес",
    "Accuracy": round(accuracy_score(y_test, rf_pred) * 100, 2),
    "Precision": round(precision_score(y_test, rf_pred, average='macro') * 100, 2),
    "Recall": round(recall_score(y_test, rf_pred, average='macro') * 100, 2),
    "F1": round(f1_score(y_test, rf_pred, average='macro') * 100, 2),
    "ROC-AUC": round(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') * 100, 2)
})

# Сохранение результатов
METRIC_DIR = BASE_DIR / "metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(METRIC_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

log.info(f"Метрики сохранены: {METRIC_DIR / 'metrics.csv'}")

import joblib

# Сохранение моделей
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(rf_model, MODEL_DIR / "random_forest.joblib")

predictions_df = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "random_forest": rf_pred
})

# Сохранение результатов
RESULTS_DIR = OUTPUT_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

# Сохранение экспериментов
import mlflow
import uuid

mlflow.set_experiment("Experiments_results")
run_name = f"RFexp{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

with mlflow.start_run(run_name=run_name):
    for block, p_dict in params.items():
        for name, value in p_dict.items():
            mlflow.log_param(f"{block}_{name}", value)

    mlflow.log_metric("rf_accuracy", accuracy_score(y_test, rf_pred))
    mlflow.log_metric("rf_f1_macro", f1_score(y_test, rf_pred, average='macro'))
    mlflow.log_metric("rf_roc_auc", roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr', average='macro'))
