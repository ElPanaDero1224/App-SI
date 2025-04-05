import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. CARGAR Y VISUALIZAR LOS DATOS
digits = load_digits()
X, y = digits.data, digits.target

# Mostrar algunas imágenes
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Etiqueta: {digits.target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 2. PREPROCESAMIENTO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. MODELOS SUPERVISADOS

## KNN
start_knn = time.time()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
end_knn = time.time()

## SVM
start_svm = time.time()
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
end_svm = time.time()

# 4. MODELOS NO SUPERVISADOS

## PCA (visualización y reducción de dimensiones)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='tab10', legend='full')
plt.title("Visualización con PCA")
plt.savefig("results/pca_visualizacion.png")
plt.show()

## K-means
start_kmeans = time.time()
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
end_kmeans = time.time()

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='tab10')
plt.title("K-means Clustering sobre PCA")
plt.savefig("results/kmeans_visualizacion.png")
plt.show()

# 5. EVALUACIÓN

def evaluar_modelo(nombre, y_true, y_pred, tiempo):
    print(f"\nModelo: {nombre}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Tiempo de ejecución: {tiempo:.4f} segundos")
    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred))

evaluar_modelo("KNN", y_test, y_pred_knn, end_knn - start_knn)
evaluar_modelo("SVM", y_test, y_pred_svm, end_svm - start_svm)

# Evaluación de clustering no supervisado: usar "accuracy aproximado"
from sklearn.metrics import adjusted_rand_score
print("\nModelo: K-means (no supervisado)")
print(f"Adjusted Rand Index (Comparado con etiquetas reales): {adjusted_rand_score(y, clusters):.4f}")
print(f"Tiempo de ejecución: {end_kmeans - start_kmeans:.4f} segundos")
