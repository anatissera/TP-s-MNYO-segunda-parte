
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Análisis de Componentes Principales
#  identifica los componentes principales de los datos -> los vectores ortogonales 
# no correlacionados entre sí y ordenados jerárquicamente que maximizan la varianza 𝜎^2 de las mediciones
# emplea SVD para calcular los componentes principales

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# La similaridad entre un par de muestras xi, xj se puede medir utilizando una función no-lineal de su distancia euclidiana
# K(xi, xj) = exp((-∥xi - xj∥_2)^2)/ (2σ^2))

# reducir d -> 
# 1) descomposición de X en sus valores singulares
# 2) reducir la dimensión de esta representación
# 3) trabajar con los vectores x proyectados al nuevo espacio reducido Z, es decir z = V^Tsubd x.
# 4) Realizar los puntos anteriores para d = 2, 6, 10, y p

# # 1. Leer el archivo CSV
# X = pd.read_csv("TP3\dataset02.csv").values

# # 2. Calcular la matriz de distancias euclidianas
# distances = squareform(pdist(X, 'euclidean'))

# # 3. Aplicar la función no lineal K(xi, xj)
# sigma = np.std(distances)
# K = np.exp(-distances**2 / (2 * sigma**2))

# # 4. Realizar la descomposición de valores singulares (SVD)
# U, S, Vt = svd(K)

# # 5. Reducir la dimensión de la representación de los datos
# # 6. Proyectar los vectores x al nuevo espacio reducido Z
# # 7. Repetir los pasos 4-6 para cada valor de d

# for d in [2, 6, 10, X.shape[1]]:
#     Z = np.dot(U[:,:d], np.diag(S[:d]))
#     print(f"Dimension reducida a {d}:")
#     print(Z)

#     # Si la dimensión es 2, podemos graficarla
#     if d == 2:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(Z[:, 0], Z[:, 1])
#         plt.title('Datos proyectados en 2D')
#         plt.xlabel('Componente principal 1')
#         plt.ylabel('Componente principal 2')
#         plt.show()
        
        
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform

# Cargar datos
# X = pd.read_csv('TP3\dataset02.csv').values
# Y = pd.read_csv('TP3\y.txt').values.ravel()

X = pd.read_csv("TP3/dataset02.csv", skiprows=1).to_numpy()
Y = pd.read_csv("TP3/y.txt").to_numpy().ravel()

# Función para calcular la similaridad
def calculate_similarity(X, sigma):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return K

sigma = 1.0  # ajustar según sea necesario

# Similaridad en el espacio original
K_X = calculate_similarity(X, sigma)

# PCA y similaridades en espacios reducidos
dims = [2, 6, 10]
similarities = {}
for d in dims:
    pca = PCA(n_components=d)
    Z_d = pca.fit_transform(X)
    K_Z_d = calculate_similarity(Z_d, sigma)
    similarities[d] = K_Z_d

# Error de predicción
errors = []
for d in dims:
    pca = PCA(n_components=d)
    Z_d = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(Z_d, Y)
    Y_pred = model.predict(Z_d)
    error = mean_squared_error(Y, Y_pred)
    errors.append((d, error))

# Error de predicción en el espacio original
model_original = LinearRegression()
model_original.fit(X, Y)
Y_pred_original = model_original.predict(X)
error_original = mean_squared_error(Y, Y_pred_original)
errors.append(('Original', error_original))

# Graficar matrices de similaridad
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.heatmap(K_X, ax=axes[0, 0], cmap='viridis')
axes[0, 0].set_title('Similaridad en el Espacio Original')

for i, d in enumerate(dims):
    row = (i + 1) // 2
    col = (i + 1) % 2
    sns.heatmap(similarities[d], ax=axes[row, col], cmap='viridis')
    axes[row, col].set_title(f'Similaridad en el Espacio Reducido d={d}')

plt.subplots_adjust(hspace=1.5, wspace=0.5)
plt.tight_layout()
plt.show()

# Visualización de los componentes principales
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, d in enumerate(dims):
    pca = PCA(n_components=d)
    pca.fit(X)
    components = pca.components_
    sns.heatmap(components, ax=axes[i], cmap='coolwarm', center=0)
    axes[i].set_title(f'Componentes Principales para d={d}')
    axes[i].set_xlabel('Dimensiones Originales')
    axes[i].set_ylabel('Componentes Principales')

plt.subplots_adjust(hspace=1)
# plt.tight_layout()
plt.show()

# Gráfico de errores de predicción
dims_labels = [str(d) for d in dims] + ['Original']
errors_values = [error for _, error in errors]

plt.figure(figsize=(10, 6))
plt.bar(dims_labels, errors_values, color='skyblue')
plt.xlabel('Dimensiones Reducidas')
plt.ylabel('Error de Predicción (MSE)')
plt.title('Errores de Predicción para Diferentes Dimensiones')
plt.show()