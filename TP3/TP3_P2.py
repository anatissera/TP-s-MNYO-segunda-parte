"""
2 Compresión de imágenes
En el archivo dataset_imagenes1.zip se encuentran n imágenes. Cada imagen es una matriz de p × p que
puede representarse como un vector x ∈ Rp∗p. A su vez, es posible armar un matriz de datos apilando los
vectores de cada imagen generando una matriz de n × (p ∗ p). Se desea aprender una representación de baja
dimensión de las imágenes mediante una descomposición en valores singulares.
1. Aprender una representación basada en Descomposición de Valores Singulares utilizando las n imágenes.
2. Visualizar en forma matricial p × p las imágenes reconstruidas luego de compresión con distintos
valores de d dimensiones ¿Qué conclusiones pueden sacar?
3. Utilizando compresión con distintos valores de d medir la similaridad entre pares de imágenes (con
alguna métrica de similaridad que decida el autor) en un espacio de baja dimensión d. Analizar cómo
la similaridad entre pares de imágenes cambia a medida que se utilizan distintos valores de d. Cuales
imágenes se encuentran cerca entre si? Alguna interpretación al respecto? Ayuda: ver de utilizar una
matriz de similaridad para visualizar todas las similaridades par-a-par juntas.
4. Dado el dataset dataset_imagenes2.zip encontrar d, el número mínimo de dimensiones a las que se
puede reducir la dimensionalidad de su representación mediante valores singulares tal que el error de
cada imagen comprimida y su original no exceda el 10% bajo la norma de Frobenius. Utilizando esta
ultima representación aprendida con el dataset 2 ¿Qué error de reconstrucción obtienen si utilizan la
misma compresión (con la misma base de d dimensiones obtenida del dataset 2) para las imagenes
dataset_imagenes1.zip?
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cada imagen es una matriz de p × p que puede representarse como un vector x ∈ Rp∗p. A su vez, es posible armar un matriz de datos apilando los vectores de cada imagen generando una matriz de n × (p ∗ p).
# tengo una carpeta con varias imagenes .jpeg
# img00.jpeg, img01.jpeg, img02.jpeg, ..., img0n.jpeg
# la carpeta de imagenes se llama datasets_imgs y esta en el mismo directorio que este script
# cargar las imagenes
import os
from PIL import Image

# cargar las imagenes
images = []
for i in range(19):
    img = Image.open(f"TP3\datasets_imgs/img{i}.jpeg")
    img = img.resize((100, 100))
    img = np.array(img)
    images.append(img)

# convertir las imagenes a un vector
images = np.array(images)
n, p, _ = images.shape
X = images.reshape(n, p*p)

# # Visualizar las imagenes
# fig, axs = plt.subplots(5, 4)
# for i, ax in enumerate(axs.flatten()):
#     if(i >= n):
#         break
#     ax.imshow(images[i], cmap="gray")
#     ax.axis("off")
# plt.show()

# Aprender una representación basada en Descomposición de Valores Singulares utilizando las n imágenes.
# Descomposición en valores singulares
U, S, Vt = np.linalg.svd(X)

# mostrar los valores singulares
fig, ax = plt.subplots()
ax.plot(S)
ax.set_title("Valores singulares")
plt.show()

# medir la similaridad entre un par de muestras xi, xj
def similarity(xi, xj, sigma=1):
    return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * sigma**2))

def error_approximation(X, X_reconstructed):
    # Error =
# ‖𝑋𝑖 − ̃ 𝑋𝑖‖𝐹 /
# ‖𝑋𝑖‖𝐹
   return (np.linalg.norm(X - X_reconstructed, ord=2) / np.linalg.norm(X, ord=2))*100

# Visualizar en forma matricial p × p las imágenes reconstruidas luego de compresión con distintos valores de d dimensiones ¿Qué conclusiones pueden sacar?
d_values = [2, 6, 10,14, 16, 50]
fig, axs = plt.subplots(2, 3)
for d, ax in zip(d_values, axs.flatten()):
    Z = np.dot(X, Vt[:d].T)
    X_reconstructed = np.dot(Z, Vt[:d])
    images_reconstructed = X_reconstructed.reshape(n, p, p)
    ax.imshow(images_reconstructed[0], cmap="gray")
    # mostrar el porcentaje de error en cada imagen
    error = error_approximation(X[0], X_reconstructed[0])
    ax.set_title(f"d={d} error={error:.2f}%")
plt.show()

# Graficar porcentaje de error de aproximacion para cada una de las imagenes a las que se les aplico SVD truncado con d = 6
d = 6
Z = np.dot(X, Vt[:d].T)
X_reconstructed = np.dot(Z, Vt[:d])
images_reconstructed = X_reconstructed.reshape(n, p, p)

# Calcular el porcentaje de error de aproximacion
error = []
for i in range(n):
    error.append(error_approximation(X[i], X_reconstructed[i]))


# Graficar en barras el error de aproximacion
fig, ax = plt.subplots()
ax.bar(range(n), error)
ax.set_title("Error de aproximación")
plt.show()


#analizar similaridad entre los distintos valores de d
# for d in d_values:
#     Z = np.dot(X, Vt[:d].T)
#     similarity_X = np.exp(-np.linalg.norm(X - X[:, None], axis=2)**2 / (2 * 1))
#     similarity_Z = np.exp(-np.linalg.norm(Z - Z[:, None], axis=2)**2 / (2 * 1))

#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(similarity_X, cmap="viridis")
#     ax[0].set_title("Similarity X")
#     ax[1].imshow(similarity_Z, cmap="viridis")
#     ax[1].set_title(f"Similarity Z d={d}")
#     plt.show()

# Utilizando compresión con distintos valores de d medir la similaridad entre pares de imágenes (con alguna métrica de similaridad que decida el autor) en un espacio de baja dimensión d. Analizar cómo la similaridad entre pares de imágenes cambia a medida que se utilizan distintos valores de d. Cuales imágenes se encuentran cerca entre si? Alguna interpretación al respecto? Ayuda: ver de utilizar una matriz de similaridad para visualizar todas las similaridades par-a-par juntas.

# # medir la similaridad entre un par de muestras xi, xj
# def similarity(xi, xj, sigma=1):
#     return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * sigma**2))

# # Similaridad entre todas las muestras
# similarity_X = np.exp(-np.linalg.norm(X - X[:, None], axis=2)**2 / (2 * 1))

# # mostrar la matriz de similaridad
# fig, ax = plt.subplots()
# ax.imshow(similarity_X, cmap="viridis")
# ax.set_title("Similarity X")
# plt.show()
