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
    img = Image.open(f"datasets_imgs/img{i}.jpeg")
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

# Visualizar en forma matricial p × p las imágenes reconstruidas luego de compresión con distintos valores de d dimensiones ¿Qué conclusiones pueden sacar?
d_values = [2, 6, 10, 50]
fig, axs = plt.subplots(2, 2)
for d, ax in zip(d_values, axs.flatten()):
    Z = np.dot(X, Vt[:d].T)
    X_reconstructed = np.dot(Z, Vt[:d])
    images_reconstructed = X_reconstructed.reshape(n, p, p)
    ax.imshow(images_reconstructed[0], cmap="gray")
    ax.axis("off")
    ax.set_title(f"d={d}")
plt.show()