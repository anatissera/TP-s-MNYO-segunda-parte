
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Análisis de Componentes Principales
#  identifica los componentes principales de los datos -> los vectores ortogonales 
# no correlacionados entre sí y ordenados jerárquicamente que maximizan la varianza 𝜎^2 de las mediciones
# emplea SVD para calcular los componentes principales

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd


# La similaridad entre un par de muestras xi, xj se puede medir utilizando una función no-lineal de su distancia euclidiana
# K(xi, xj) = exp((-∥xi - xj∥_2)^2)/ (2σ^2))

# reducir d -> 
# 1) descomposición de X en sus valores singulares
# 2) reducir la dimensión de esta representación
# 3) trabajar con los vectores x proyectados al nuevo espacio reducido Z, es decir z = V^Tsubd x.
# 4) Realizar los puntos anteriores para d = 2, 6, 10, y p

# 1. Leer el archivo CSV
X = pd.read_csv("TP3\dataset02.csv").values

# 2. Calcular la matriz de distancias euclidianas
distances = squareform(pdist(X, 'euclidean'))

# 3. Aplicar la función no lineal K(xi, xj)
sigma = np.std(distances)
K = np.exp(-distances**2 / (2 * sigma**2))

# 4. Realizar la descomposición de valores singulares (SVD)
U, S, Vt = svd(K)

# 5. Reducir la dimensión de la representación de los datos
# 6. Proyectar los vectores x al nuevo espacio reducido Z
# 7. Repetir los pasos 4-6 para cada valor de d

for d in [2, 6, 10, X.shape[1]]:
    Z = np.dot(U[:,:d], np.diag(S[:d]))
    print(f"Dimension reducida a {d}:")
    print(Z)

    # Si la dimensión es 2, podemos graficarla
    if d == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(Z[:, 0], Z[:, 1])
        plt.title('Datos proyectados en 2D')
        plt.xlabel('Componente principal 1')
        plt.ylabel('Componente principal 2')
        plt.show()