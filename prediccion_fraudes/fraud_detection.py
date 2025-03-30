# -*- coding: utf-8 -*-
"""
Predicción de Fraudes con Modelos Supervisados (Regresión Logística, XGBoost) y No Supervisados (Isolation Forest).
Dataset: Credit Card Fraud Detection (Kaggle).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# PASO 1: Cargar y explorar datos
# ------------------------
# Descargar el dataset de Kaggle manualmente y colocarlo en la carpeta 'data'
data = pd.read_csv('creditcard.csv')

# Ver distribución de clases
print("Distribución de clases:\n", data['Class'].value_counts())
sns.countplot(x='Class', data=data)
plt.title("Distribución de Fraudes vs No Fraudes")
plt.savefig("resultados_fraudes/class_distribution.png")
plt.show()

# ------------------------
# PASO 2: Preprocesamiento
# ------------------------
# Separar características (X) y etiquetas (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Balancear datos (undersampling para evitar overfitting)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# ------------------------
# PASO 3: Modelo Supervisado (Regresión Logística)
# ------------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n--- Regresión Logística ---")
print(classification_report(y_test, y_pred_lr))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_lr))

# ------------------------
# PASO 4: Modelo No Supervisado (Isolation Forest)
# ------------------------
# Isolation Forest detecta anomalías (fraudes = outliers)
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]  # Convertir -1 (outlier) a 1 (fraude)

print("\n--- Isolation Forest ---")
print(classification_report(y_test, y_pred_iso))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_iso))

# ------------------------
# PASO 5: Visualización
# ------------------------
# Matriz de confusión para Regresión Logística
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión - Regresión Logística")
plt.savefig("resultados_fraudes/cm_logistic_regression.png")
plt.show()