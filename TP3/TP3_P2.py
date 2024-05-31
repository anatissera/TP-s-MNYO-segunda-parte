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
plt.rcParams["font.family"] = "serif"
# Cada imagen es una matriz de p × p que puede representarse como un vector x ∈ Rp∗p. A su vez, es posible armar un matriz de datos apilando los vectores de cada imagen generando una matriz de n × (p ∗ p).
# tengo una carpeta con varias imagenes .jpeg
# img00.jpeg, img01.jpeg, img02.jpeg, ..., img0n.jpeg
# la carpeta de imagenes se llama datasets_imgs y esta en el mismo directorio que este script
# cargar las imagenes
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# cargar las imagenes
images = []
for i in range(0,19):
    img = Image.open(f"TP3\datasets_imgs/img{i}.jpeg")
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

# 

def plot_singular_values(S):
    # Valores singulares {𝜎𝑖}19
    # 𝑖=1 de 𝐴 en escala semilogarítmica
    # respecto al eje de ordenadas. Se observa que se hallan ordenados
    # de forma decreciente y que el primer valor singular 𝜎1 duplica
    # al siguiente valor singular 𝜎2, frente a un menor decrecimiento
    # valor a valor para 𝑖 subsiguientes.

    fig, ax= plt.subplots()
    ax.plot(S, marker="o",color="slateblue")
    ax.set_yscale("log")
    ax.set_title("Valores singulares $\\sigma_i$", fontsize=12)
    ax.set_xlabel("i", fontsize=12)
    ax.set_ylabel("Valor singular $sigma_i$", fontsize=12)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
    ax.tick_params(axis='y', which='both', labelsize=10)
    plt.show()
    #Proporción de la suma acumulada de {𝜎𝑖}19
    # 𝑖=1 de 𝐴
    # respecto de la desviación 𝜎 total (ver 12). Se representa el
    # aporte de información de las primeras 𝑟 componentes de la
    # descomposición SVD de 𝑋. En particular, se expone que,
    # tomando las primeras 5 componentes 𝐮𝑖𝜎𝑖𝐯⊤, se concentra el
    # 50% de la información no redundante de 𝑋.

    # Calcular la proporción de la suma acumulada de los valores singulares
    cumsum = np.cumsum(S)
    cumsum /= cumsum[-1]


    fig, axs = plt.subplots()

    axs.plot(cumsum, marker="o", color="slateblue")
    axs.set_title("Proporción de la suma acumulada")
    plt.show()
    # mostrar los valores singulares
# def plot_singular_values(S):
#     # Valores singulares {𝜎𝑖}19
#     # 𝑖=1 de 𝐴 en escala semilogarítmica
#     # respecto al eje de ordenadas. Se observa que se hallan ordenados
#     # de forma decreciente y que el primer valor singular 𝜎1 duplica
#     # al siguiente valor singular 𝜎2, frente a un menor decrecimiento
#     # valor a valor para 𝑖 subsiguientes.
#     fig, ax = plt.subplots()
#     ax.plot(S)
#     ax.set_title("Valores singulares")
#     plt.show()

plot_singular_values(S)

# medir la similaridad entre un par de muestras xi, xj
def similarity(xi, xj, sigma=1):
    return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * sigma**2))

def error_approximation(X, X_reconstructed):
#    error respecto a las imágenes originales que induce su reconstrucción a partir de los primeros 8 autovectores {𝐮𝑖}8 𝑖=1 (primera mitad de la base de autovectores 𝑈𝐴) en contraste con el mismo experimento llevado a cabo con las últimas 8 componentes de 𝑈𝐴 {𝐮𝑖}16 𝑖=9 (segunda mitad de la base de autovectores)
    # Error =
# ‖𝑋𝑖 − ̃ 𝑋𝑖‖𝐹 /
# ‖𝑋𝑖‖𝐹
    return np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X) * 100

# Visualizar en forma matricial p × p las imágenes reconstruidas luego de compresión con distintos valores de d dimensiones ¿Qué conclusiones pueden sacar?

def plot_images_reconstructed(X, Vt, d_values):
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

def plot_images_reconstructed_last_values(X, Vt, d_values):
    fig, axs = plt.subplots(2, 3)
    for d, ax in zip(d_values, axs.flatten()):
        Z = np.dot(X, Vt[d:].T)
        X_reconstructed = np.dot(Z, Vt[d:])
        images_reconstructed = X_reconstructed.reshape(n, p, p)
        ax.imshow(images_reconstructed[0], cmap="gray")
        # mostrar el porcentaje de error en cada imagen
        error = error_approximation(X[0], X_reconstructed[0])
        ax.set_title(f"d={d} error={error:.2f}%")
    plt.show()

d_values = [2, 6, 10, 14, 16, 19]
plot_images_reconstructed(X, Vt, d_values)

plot_images_reconstructed_last_values(X, Vt, d_values)

def compare_last_values_vs_first(X, Vt):
    # comparar errores entre usar las primeras 10 componentes y las últimas 9
    errors_first = []
    errors_last = []
    d = 10
    for i in range(n):
        Z = np.dot(X[i], Vt[:d].T)
        X_reconstructed = np.dot(Z, Vt[:d])
        errors_first.append(error_approximation(X[i], X_reconstructed))

        Z = np.dot(X[i], Vt[d:].T)
        X_reconstructed = np.dot(Z, Vt[d:])
        errors_last.append(error_approximation(X[i], X_reconstructed))

    fig, ax = plt.subplots()
    ax.bar(np.arange(n), errors_last, label="Últimas 9 componentes", color="slateblue")
    ax.bar(np.arange(n), errors_first, label="Primeras 10 componentes", color="salmon")
    ax.set_title("Error de aproximación")
    ax.legend()
    plt.show()

compare_last_values_vs_first(X, Vt)


# Graficar porcentaje de error de aproximacion para cada una de las imagenes a las que se les aplico SVD truncado con d = 6
def plot_error_approximation(X, Vt, d1, d2):
    fig, ax = plt.subplots()
    error1 = []
    error2 = []
    for i in range(n):
        Z = np.dot(X[i], Vt[:d1].T)
        X_reconstructed = np.dot(Z, Vt[:d1])
        error1.append(error_approximation(X[i], X_reconstructed))

        Z = np.dot(X[i], Vt[:d2].T)
        X_reconstructed = np.dot(Z, Vt[:d2])
        error2.append(error_approximation(X[i], X_reconstructed))

    ax.bar(np.arange(n), error1, label=f"d={d1}", color="salmon")
    ax.bar(np.arange(n), error2, label=f"d={d2}", color="slateblue")
    ax.set_title("Error de aproximación")
    ax.legend()
    plt.show()

plot_error_approximation(X, Vt, 4,8) 
# intente copiar el de segundo pero me parece que no es el mismo


# usar el dataset 2
# cargar las imagenes
images = []
for i in range(8):
    img = Image.open(f"TP3\datasets_imgs_02/img0{i}.jpeg")
    img = np.array(img)
    images.append(img)

# convertir las imagenes a un vector
images = np.array(images)
n, p, _ = images.shape
X = images.reshape(n, p*p)

# Descomposición en valores singulares
U, S, Vt = np.linalg.svd(X)

def plot_minimum_dimension(X, error_threshold):
    # Dado el dataset dataset_imagenes2.zip encontrar d, el número mínimo de dimensiones a las que se
    # puede reducir la dimensionalidad de su representación mediante valores singulares tal que el error de
    # cada imagen comprimida y su original no exceda el 10% bajo la norma de Frobenius. Utilizando esta
    # ultima representación aprendida con el dataset 2 ¿Qué error de reconstrucción obtienen si utilizan la
    # misma compresión (con la misma base de d dimensiones obtenida del dataset 2) para las imagenes
    # dataset_imagenes1.zip?
    
    # hacer un grafico de barras, en el eje Y que muestre dimensiones, y en el eje X el numero de imagen. Con esto vamos a ver el numero minimo de dimensiones para que cada imagen tenga un error menor al 10%
    fig, ax = plt.subplots()
    d_values = np.arange(1, X.shape[1])
    for i in range(8):
        error = []
        for d in d_values:
            Z = np.dot(X[i], Vt[:d].T)
            X_reconstructed = np.dot(Z, Vt[:d])
            error.append(error_approximation(X[i], X_reconstructed))
            if(error[-1] < error_threshold):
                break
        ax.bar(i, d, color="slateblue")
    ax.set_title("Número mínimo de dimensiones")
    plt.show()

plot_minimum_dimension(X, 10)




# Utilizando compresión con distintos valores de d medir la similaridad entre pares de imágenes (con
# alguna métrica de similaridad que decida el autor) en un espacio de baja dimensión d. Analizar cómo
# la similaridad entre pares de imágenes cambia a medida que se utilizan distintos valores de d. Cuales
# imágenes se encuentran cerca entre si? Alguna interpretación al respecto? Ayuda: ver de utilizar una
# matriz de similaridad para visualizar todas las similaridades par-a-par juntas.

# funcion para reducir la dimension y calcular la matriz de similaridad
def similarity_matrix(U, S, d):
    U_reduced = U[:, :d]
    S_reduced = np.diag(S[:d])
    Z = np.dot(U_reduced, S_reduced)
    similarity_matrixe = cosine_similarity(Z)
    return similarity_matrixe

# Visualizar la matriz de similitud para disitntos valores de d
def plot_similarity_matrix(U, S, d_values):
    fig, axs = plt.subplots(2, 3)
    for d, ax in zip(d_values, axs.flatten()):
        similarity_matrixe = similarity_matrix(U, S, d)
        ax.imshow(similarity_matrixe, cmap="viridis")
        ax.set_title(f"d={d}")
    plt.show()

d_values = [2, 6, 10, 14, 16, 50]
plot_similarity_matrix(U, S, d_values)
