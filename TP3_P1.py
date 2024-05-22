# Trabajo Práctico 3 - SVD y reducción de la dimensionalidad

"""

1 Reducción de la dimensionalidad y Cuadrados Mínimos
En el archivo dataset.csv se encuentra el dataset X. Este contiene un conjunto de n muestras que fueron
medidas a través de un sensor
{x1, x2, . . . , xi, . . . , xn}
con xi ∈ Rp (X es por lo tanto una matriz de nxp dimensiones). Si bien el conjunto tiene, a priori, dimensión alta, es de interés entender visualmente como se distribuyen las muestras. Suponemos que las muestras no se distribuyen uniformemente en el espacio Rp, por lo que podremos encontrar grupos de muestras (clusters) con alta similaridad entre sí. La similaridad entre un par de muestras xi, xj se puede medir utilizando una función no-lineal de su distancia euclidiana
K (xi, xj) = exp(-∥xi - xj∥_2)^2/ (2σ^2)),
para algún valor de σ.
Como la dimensionalidad inicial del dataset es muy alta y se supone que algunas dimensiones son mas
ruidosas que otras en las muestras, va a ser conveniente trabajar en un espacio de dimensión reducida d.
Para hacer esto hay que realizar una descomposición de X en sus valores singulares, reducir la dimensión de esta representación, y luego trabajar con los vectores x proyectados al nuevo espacio reducido Z, es decir
z = (V^T)_d x. Realizar los puntos anteriores para d = 2, 6, 10, y p. ¿Para qué elección de d resulta más conveniente hacer el análisis? ¿Cómo se conecta esto con los valores singulares de X? ¿Qué conclusiones puede sacar al respecto?

1. Determinar la similaridad par-a-par entre muestras en el espacio de dimension X y en el espacio
de dimensión reducida d para distintos valores de d utilizando PCA. Comparar estas medidas de
similaridad. Ayuda: ver de utilizar una matriz de similaridad para visualizar todas las similaridades
par-a-par juntas.

2. De las p dimensiones originales del dataset, cuales son las mas representativas con respecto a las
dimensiones d obtenidas por SVD? Indicar que dimensiones originales del conjunto p son las mas
importantes y el método utilizado para determinarlas.

3. Los datosX vienen acompañados de una variable dependiente respuesta o etiquetas llamada Y (archivo
y.txt) estructurada como un vector nx1 para cada muestra. Queremos encontrar el vector β y modelar
linealmente el problema que minimice la norma 
∥Xβ - y∥_2
de manera tal de poder predecir con Xβ = ŷ lo mejor posible a las etiquetas y, es decir, minimizar el
error de predicción. Usando PCA, que dimensión d mejora la predicción? Cuales muestras son las de
mejor predicción con el mejor modelo? Resolviendo el problema de cuadrados mínimos en el espacio
original X, que peso se le asigna a cada dimensión original si observamos el vector β?
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
# X = pd.read_csv("dataset02.csv").to_numpy() primer linea cuenta las columnas, saltarla
X = pd.read_csv("dataset02.csv", skiprows=1).to_numpy()
Y = pd.read_csv("y.txt").to_numpy()

# Graficar las muestras de la primer columna de X
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap="viridis")
ax.set_title("Dataset")
plt.show()

# Suponemos que las muestras no se distribuyen uniformemente en el espacio Rp, por lo que podremos encontrar grupos de muestras (clusters) con alta similaridad entre sí. La similaridad entre un par de muestras xi, xj se puede medir utilizando una función no-lineal de su distancia euclidiana
# K (xi, xj) = exp(-∥xi - xj∥_2)^2/ (2σ^2)),

# medir la similaridad entre un par de muestras xi, xj
def similarity(xi, xj, sigma=1):
    return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * sigma**2))

# Similaridad entre todas las muestras
similarity_X = np.exp(-np.linalg.norm(X - X[:, None], axis=2)**2 / (2 * 1))

# mostrar la matriz de similaridad
fig, ax = plt.subplots()
ax.imshow(similarity_X, cmap="viridis")
ax.set_title("Similarity X")
plt.show()





# lo q me tira el copiloto y no anda pero lo guardo por las dudas:
"""
# 1. Reducción de la dimensionalidad
# 1.1 Descomposición de X en sus valores singulares
U, S, Vt = np.linalg.svd(X)

# 1.2 Reducción de la dimensión de la representación
d_values = [2, 6, 10, X.shape[1]]
for d in d_values:
    # Reducción de la dimensión de la representación
    Z = np.dot(X, Vt[:d].T)

    # 1.3 Proyección de los vectores x al nuevo espacio reducido Z
    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1], c=Y.flatten(), cmap="viridis")
    ax.set_title(f"PCA d={d}")
    plt.show()

    # 1. Determinar la similaridad par-a-par entre muestras en el espacio de dimension X y en el espacio
    # de dimensión reducida d para distintos valores de d utilizando PCA. Comparar estas medidas de
    # similaridad. Ayuda: ver de utilizar una matriz de similaridad para visualizar todas las similaridades
    # par-a-par juntas.
    pca = PCA(n_components=d)
    Z = pca.fit_transform(X)
    similarity_X = np.exp(-np.linalg.norm(X - X[:, None], axis=2)**2 / (2 * 1))
    similarity_Z = np.exp(-np.linalg.norm(Z - Z[:, None], axis=2)**2 / (2 * 1))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(similarity_X, cmap="viridis")
    ax[0].set_title("Similarity X")
    ax[1].imshow(similarity_Z, cmap="viridis")
    ax[1].set_title("Similarity Z")
    plt.show()

    # 2. De las p dimensiones originales del dataset, cuales son las mas representativas con respecto a las
    # dimensiones d obtenidas por SVD? Indicar que dimensiones originales del conjunto p son las mas
    # importantes y el método utilizado para determinarlas.
    pca = PCA(n_components=d)
    pca.fit(X)
    print(f"PCA components d={d}: {pca.components_}")

    # 3. Los datosX vienen acompañados de una variable dependiente respuesta o etiquetas llamada Y (archivo
    # y.txt) estructurada como un vector nx1 para cada muestra. Queremos encontrar el vector β y modelar
    # linealmente el problema que minimice la norma 
    # ∥Xβ - y∥_2
    # de manera tal de poder predecir con Xβ = ŷ lo mejor posible a las etiquetas y, es decir, minimizar el
    # error de predicción. Usando PCA, que dimensión d mejora la predicción? Cuales muestras son las de
    # mejor predicción con el mejor modelo? Resolviendo el problema de cuadrados mínimos en el espacio
    # original X, que peso se le asigna a cada dimensión original si observamos el vector β?
    pca = PCA(n_components=d)
    Z = pca.fit_transform(X)
    lr = LinearRegression()
    lr.fit(Z, Y)
    Y_pred = lr.predict(Z)
    mse = mean_squared_error(Y, Y_pred)
    print(f"MSE PCA d={d}: {mse}")

    lr = LinearRegression()
    lr.fit(X, Y)
    Y_pred = lr.predict(X)
    mse = mean_squared_error(Y, Y_pred)
    print(f"MSE X d={d}: {mse}")

    print(f"PCA components d={d}: {lr.coef_}")
    print(f"X components d={d}: {lr.coef_}")

    # ¿Para qué elección de d resulta más conveniente hacer el análisis? ¿Cómo se conecta esto con los valores singulares de X? ¿Qué conclusiones puede sacar al respecto?
    # La elección de d más conveniente es 10, ya que minimiza el error de predicción. Los valores singulares de X nos permiten reducir la dimensión de la representación de X, y las dimensiones más importantes son las que tienen mayor peso en el vector β.

"""