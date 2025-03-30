# -*- coding: utf-8 -*-
"""
Clasificador de Texto con KNN (supervisado) y K-means (no supervisado).
Dataset: 20 Newsgroups
"""

# Importar bibliotecas
from sklearn.datasets import fetch_20newsgroups  # Para cargar el dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # Para convertir texto a vectores numéricos
from sklearn.neighbors import KNeighborsClassifier  # Modelo KNN
from sklearn.cluster import KMeans  # Modelo K-means
from sklearn.metrics import accuracy_score, classification_report  # Métricas de evaluación
import matplotlib.pyplot as plt  # Para gráficos
from sklearn.decomposition import PCA  # Para reducir dimensionalidad y visualizar clusters
import seaborn as sns  # Para gráficos más atractivos

# ------------------------
# PASO 1: Cargar el dataset
# ------------------------
# Seleccionamos 4 categorías para simplificar el ejemplo (puedes cambiarlas)
categories = ['rec.sport.hockey', 'sci.space', 'talk.religion.misc', 'comp.graphics']

# Descargar datos de entrenamiento y prueba
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Imprimir ejemplo de un documento y su categoría
print("\nEjemplo de documento:")
print(newsgroups_train.data[0][:200])  # Primeros 200 caracteres del primer documento
print("\nCategoría:", newsgroups_train.target_names[newsgroups_train.target[0]])

# ------------------------
# PASO 2: Preprocesamiento (Convertir texto a números)
# ------------------------
# Usamos TF-IDF para vectorizar el texto (mejor que CountVectorizer para KNN)
vectorizer = TfidfVectorizer(stop_words='english')  # Ignora palabras comunes (the, and, etc.)
X_train = vectorizer.fit_transform(newsgroups_train.data)  # Aprende vocabulario y transforma datos de entrenamiento
X_test = vectorizer.transform(newsgroups_test.data)  # Transforma datos de prueba (sin aprender vocabulario nuevamente)

# Etiquetas (targets)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# ------------------------
# PASO 3: Modelo Supervisado (KNN)
# ------------------------
# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Usamos 5 vecinos (puedes ajustar este valor)
knn.fit(X_train, y_train)

# Predecir en datos de prueba
y_pred_knn = knn.predict(X_test)

# Evaluar el modelo
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\n--- Resultados de KNN ---")
print(f"Precisión: {accuracy_knn:.2f}")
print(classification_report(y_test, y_pred_knn, target_names=newsgroups_train.target_names))

# ------------------------
# PASO 4: Modelo No Supervisado (K-means)
# ------------------------
# Crear y entrenar K-means (usamos 4 clusters porque hay 4 categorías)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_train)

# Obtener clusters asignados a cada documento
clusters = kmeans.labels_

# Evaluación (no hay etiquetas verdaderas en no supervisado, pero comparamos con las categorías reales)
# Nota: En problemas reales, esto no es posible (solo para análisis)
print("\n--- Resultados de K-means ---")
print("Comparación manual con categorías reales:")
for i in range(4):
    print(f"Cluster {i}: Mayoría es categoría {newsgroups_train.target_names[y_train[clusters == i][0]]}")

# ------------------------
# PASO 5: Visualización (PCA + K-means)
# ------------------------
# Reducir dimensionalidad a 2D para graficar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train.toarray())  # Convertir matriz dispersa a densa

# Crear DataFrame para Seaborn
import pandas as pd
df = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
df['Cluster'] = clusters
df['Categoría Real'] = y_train

# Graficar clusters vs categorías reales
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Componente 1', y='Componente 2', hue='Cluster', palette='viridis')
plt.title("Clusters de K-means")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Componente 1', y='Componente 2', hue='Categoría Real', palette='tab10')
plt.title("Categorías Reales")

plt.tight_layout()
plt.savefig("clusters_vs_reales.png")  # Guardar gráfico
plt.show()