import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings("ignore")

# Чтение файлов, обработка датасетов
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df1 = pd.read_csv(DATA_DIR / "thyroidDF.csv")
df2 = pd.read_csv(DATA_DIR / "hypothyroid.csv")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

# Заполнение пустых значений
mode = X_train[categorical_cols].mode().iloc[0]
X_train[categorical_cols] = X_train[categorical_cols].fillna(mode)
X_test[categorical_cols] = X_test[categorical_cols].fillna(mode)

imputer = KNNImputer(n_neighbors=5, weights='distance')

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
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
results = []

# Логистическая регрессия
print("Модель логистической регрессии")

start_time = time.perf_counter()

log_model = LogisticRegression(penalty='l1', C=0.1, max_iter=100, solver='saga' , random_state=42, class_weight='balanced',  n_jobs=-1)

log_model.fit(X_train, y_train)
end_time = time.perf_counter()
log_pred = log_model.predict(X_test)

print(f"Время обучения: {end_time - start_time} секунд")

print("Accuracy: ", accuracy_score(y_test, log_pred))
print("Precision", precision_score(y_test, log_pred, average='macro'))
print("Recall", recall_score(y_test, log_pred, average='macro'))
print("F1-score: ", f1_score(y_test, log_pred, average='macro'))

y_proba = log_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
print("ROC-AUC: ", roc_auc)

print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

results.append({
    "Model": "Логистическая регрессия",
    "Accuracy": round(accuracy_score(y_test, log_pred) * 100, 2),
    "Precision": round(precision_score(y_test, log_pred, average='macro') * 100, 2),
    "Recall": round(recall_score(y_test, log_pred, average='macro') * 100, 2),
    "F1": round(f1_score(y_test, log_pred, average='macro') * 100, 2),
    "ROC-AUC": round(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') * 100, 2)
})

# Случайный лес
print("Модель случайного леса")

start_time = time.perf_counter()

rf_model = RandomForestClassifier(max_depth=15, max_features='sqrt', n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)

rf_model.fit(X_train, y_train)
end_time = time.perf_counter()
rf_pred = rf_model.predict(X_test)

print(f"Время обучения: {end_time - start_time} секунд")

print("Accuracy: ", accuracy_score(y_test, rf_pred))
print("Precision: ", precision_score(y_test, rf_pred, average='macro'))
print("Recall: ", recall_score(y_test, rf_pred, average='macro'))
print("F1-score: ", f1_score(y_test, rf_pred, average='macro'))

y_proba = rf_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

print("ROC-AUC: ", roc_auc)

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

print(f"\nМетрики сохранены: {METRIC_DIR / 'metrics.csv'}")

import joblib

# Сохранение моделей
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(log_model, MODEL_DIR / "logistic_regression.joblib")
joblib.dump(rf_model, MODEL_DIR / "random_forest.joblib")

predictions_df = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "logistic_regression": log_pred,
    "random_forest": rf_pred
})

# Сохранение результатов
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")