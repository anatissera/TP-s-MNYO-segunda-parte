# Punto 2 - Compresion de imágenes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
plt.rcParams["font.family"] = "serif"
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
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

U, S, Vt = np.linalg.svd(X)

def plot_singular_values(S):
    # Graficar los valores singulares
    fig, ax= plt.subplots()
    ax.plot(S, marker="o",markersize=5, color="slateblue")
    ax.set_yscale("log")
    ax.set_title("Valores singulares $\\sigma_i$", fontsize=15)
    ax.set_xlabel("i", fontsize=15)
    ax.set_ylabel("$\\sigma_i$", fontsize=13)
    ax.set_xticks(np.arange(0, 19, 1))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
    ax.tick_params(axis='y', which='both', labelsize=12)
    ax.tick_params(axis='x', which='both', labelsize=12)
    plt.show()

    # Calcular la proporción de la suma acumulada de los valores singulares
    cumsum = np.cumsum(S)
    cumsum /= cumsum[-1]

    fig, axs = plt.subplots()
    axs.plot(cumsum, marker="o",markersize=5, color="slateblue", label="Proporción de la suma acumulada")
    axs.hlines(y=0.725, xmin=0, xmax=9, color="salmon", linestyle="--")
    axs.vlines(x=9, ymin=0, ymax=0.725, color="salmon", linestyle="--")
    axs.set_xticks(np.arange(0, 19, 1))
    axs.set_yticks(np.arange(0, 1.1, 0.1))
    axs.set_xlim(-0.5, 19)
    axs.set_ylim(0.13, 1.03)
    axs.set_xlabel("i")
    axs.set_ylabel("Proporción")
    axs.scatter (9, 0.725, color="salmon", label="d=9 (72.5%)", zorder=10)
    axs.legend()
    plt.show()

plot_singular_values(S)

# medir la similaridad entre un par de muestras xi, xj
def similarity(xi, xj, sigma=1):
    return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * sigma**2))

def error_approximation(X, X_reconstructed):
    return np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X) * 100

def plot_images_reconstructed(X, Vt, d_values):
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for d, ax in zip(d_values, axs.flatten()):
        Z = np.dot(X, Vt[:d].T)
        X_reconstructed = np.dot(Z, Vt[:d])
        images_reconstructed = X_reconstructed.reshape(n, p, p)
        ax.imshow(images_reconstructed[11], cmap="gray")
        error = error_approximation(X[11], X_reconstructed[11])
        ax.axis("off")
        ax.set_title(f"d={d} error={error:.2f}%")
    plt.show()

def plot_images_reconstructed_last_values(X, Vt, d_values):
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for d, ax in zip(d_values, axs.flatten()):
        Z = np.dot(X, Vt[d:].T)
        X_reconstructed = np.dot(Z, Vt[d:])
        images_reconstructed = X_reconstructed.reshape(n, p, p)
        ax.imshow(images_reconstructed[11], cmap="gray")
        error = error_approximation(X[11], X_reconstructed[11])
        ax.axis("off")
        ax.set_title(f"d={d} error={error:.1f}%")
    plt.show()

d_values = [2, 6, 12, 19]
plot_images_reconstructed(X, Vt, d_values)

plot_images_reconstructed_last_values(X, Vt, d_values)

def compare_last_values_vs_first(X, Vt):
    # comparar errores entre usar las primeras 10 componentes y las últimas 9
    errors_first = []
    errors_last = []
    d = 9
    for i in range(n):
        Z = np.dot(X[i], Vt[:d].T)
        X_reconstructed = np.dot(Z, Vt[:d])
        errors_first.append(error_approximation(X[i], X_reconstructed))

        Z = np.dot(X[i], Vt[d:].T)
        X_reconstructed = np.dot(Z, Vt[d:])
        errors_last.append(error_approximation(X[i], X_reconstructed))

    fig, ax = plt.subplots()
    ax.bar(np.arange(n), errors_last, label="Últimas 10 componentes", color="slateblue")
    ax.bar(np.arange(n), errors_first, label="Primeras 9 componentes", color="salmon")
    ax.set_title("Error de aproximación")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_xlabel("Número de imagen")
    ax.set_ylabel("Error")
    ax.set_yticklabels([f"{int(error)}%" for error in ax.get_yticks()])
    ax.legend(loc="lower left")
    plt.show()

compare_last_values_vs_first(X, Vt)

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

    ax.bar(np.arange(n), error1, label=f"d={d1}", color="slateblue")
    ax.bar(np.arange(n), error2, label=f"d={d2}", color="salmon")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_xlabel("Número de imagen")
    ax.set_ylabel("Error")
    ax.set_yticklabels([f"{int(error)}%" for error in ax.get_yticks()])
    ax.set_title("Error de aproximación")
    ax.legend(loc="lower left")
    plt.show()

plot_error_approximation(X, Vt, 4,8)

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
        # agregar algo de decoracion para que no sea tan aburrido el grafico
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(np.arange(8))
    ax.set_xlabel("Número de imagen")
    ax.set_ylabel("dimensiones")
        # graficar una linea que indique la dimension mas alta
    ax.hlines(y=8, xmin=-1, xmax=8, color="salmon", linestyle="--")
    ax.set_title("Número mínimo de dimensiones")
    plt.show()

    # mostrar 4 imagenes del dataset 2 de ejemplo
    # usar el dataset 2
    # cargar las imagenes
    images = []
    for i in range(8):
        img = Image.open(f"TP3\datasets_imgs_02/img0{i}.jpeg")
        img = np.array(img)
        images.append(img)

    # convertir las imagenes a un vector
    images = np.array(images)

    fig, axs = plt.subplots(2, 4)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    plt.show()

    # usar la representacion aprendida con el dataset 2
    # usar esta compresion para las imagenes del dataset 1
    #cargar imagenes del dataset1
    d = 8
    images = []
    for i in range(19):
        img = Image.open(f"TP3\datasets_imgs/img{i}.jpeg")
        img = np.array(img)
        images.append(img)

    # convertir las imagenes a un vector
    images = np.array(images)
    n, p, _ = images.shape
    X = images.reshape(n, p*p)

    # calcular el error de reconstruccion utilizando la compresion aprendida con el dataset 2
    errors = []
    for i in range(n):
        Z = np.dot(X[i], Vt[:d].T)
        X_reconstructed = np.dot(Z, Vt[:d])
        errors.append(error_approximation(X[i], X_reconstructed))

    fig, ax = plt.subplots()
    ax.bar(np.arange(n), errors, color="slateblue")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(n))
    ax.set_xlabel("Número de imagen")
    ax.set_ylabel("Error")
    ax.set_yticklabels([f"{int(error)}%" for error in ax.get_yticks()])
    ax.set_title("Error de reconstrucción")
    plt.show()

    # graficar las imagenes reconstruidas de tres imagenes del dataset 1, y arriba sus originales
    fig, axs = plt.subplots(2, 3)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    axs[0,0].imshow(images[1], cmap="gray")
    axs[0,0].set_title("Imagen 1")
    axs[0,0].axis("off")

    Z = np.dot(X[1], Vt[:d].T)
    X_reconstructed = np.dot(Z, Vt[:d])
    image_reconstructed = X_reconstructed.reshape(p, p)
    axs[1,0].imshow(image_reconstructed, cmap="gray")
    axs[1,0].set_title("Reconstruida")
    axs[1,0].axis("off")

    axs[0,1].imshow(images[16], cmap="gray")
    axs[0,1].set_title("Imagen 16")
    axs[0,1].axis("off")

    Z = np.dot(X[16], Vt[:d].T)
    X_reconstructed = np.dot(Z, Vt[:d])
    image_reconstructed = X_reconstructed.reshape(p, p)
    axs[1,1].imshow(image_reconstructed, cmap="gray")
    axs[1,1].set_title("Reconstruida")
    axs[1,1].axis("off")

    axs[0,2].imshow(images[18], cmap="gray")
    axs[0,2].set_title("Imagen 18")
    axs[0,2].axis("off")

    Z = np.dot(X[18], Vt[:d].T)
    X_reconstructed = np.dot(Z, Vt[:d])
    image_reconstructed = X_reconstructed.reshape(p, p)
    axs[1,2].imshow(image_reconstructed, cmap="gray")
    axs[1,2].set_title("Reconstruida")
    axs[1,2].axis("off")

    plt.show()

plot_minimum_dimension(X, 10)
images = []
for i in range(19):
    img = Image.open(f"TP3\datasets_imgs/img{i}.jpeg")
    img = np.array(img)
    images.append(img)

# convertir las imagenes a un vector
images = np.array(images)
n, p, _ = images.shape
X = images.reshape(n, p*p)

U, S, Vt = np.linalg.svd(X)

# funcion para reducir la dimension y calcular la matriz de similaridad
def similarity_matrix(U, S, d):
    U_reduced = U[:, :d]
    S_reduced = np.diag(S[:d])
    Z = np.dot(U_reduced, S_reduced)
    similarity_matrixe = cosine_similarity(Z)
    return similarity_matrixe

# Visualizar la matriz de similitud para disitntos valores de d
def plot_similarity_matrix(U, S, d_values):
    fig, axs = plt.subplots(2, 2)
    for d, ax in zip(d_values, axs.flatten()):
        similarity_matrixe = similarity_matrix(U, S, d)
        ax.imshow(similarity_matrixe, cmap="viridis")
        ax.set_title(f"d={d}")
        
    #agregar una barra de color para que se pueda interpretar mejor
    fig.colorbar(axs[0, 0].imshow(similarity_matrixe, cmap="viridis"), ax=axs, orientation='vertical')
    plt.show()

d_values = [2, 6, 12, 19]
plot_similarity_matrix(U, S, d_values)

def comparar_imagenes_similares():
    fig, axs = plt.subplots(2, 3)
    plt.subplots_adjust(wspace=0, hspace=0.2)
    # mostrar imagen 1 y 17 con d=19
    d = 19

    images = []
    for i in range(19):
        img = Image.open(f"TP3\datasets_imgs/img{i}.jpeg")
        img = np.array(img)
        images.append(img)

    # convertir las imagenes a un vector
    images = np.array(images)
    n, p, _ = images.shape
    X = images.reshape(n, p*p)

    U, S, Vt = np.linalg.svd(X)
    U_reduced = U[:, :d]
    S_reduced = np.diag(S[:d])
    Z = np.dot(U_reduced, S_reduced)
    X_reconstructed = np.dot(Z, Vt[:d])
    image1 = X_reconstructed[1].reshape(p, p)
    image17 = X_reconstructed[17].reshape(p, p)

    image18 = X_reconstructed[18].reshape(p, p)
    image10 = X_reconstructed[10].reshape(p, p)

    # mostrar las imagenes 
    axs[0, 0].imshow(image1, cmap="gray")
    axs[0, 0].set_title("Imagen 1")
    axs[0,0].axis('off')
    axs[0, 1].imshow(image17, cmap="gray")
    axs[0, 1].set_title("Imagen 17")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(image10, cmap="gray")
    axs[1, 0].set_title("Imagen 10")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(image18, cmap="gray")
    axs[1, 1].set_title("Imagen 18")
    axs[1, 1].axis('off')

    #superponer las imagenes 10 y 18 en una misma imagen con un 50% de opacidad
    image10_18 = image10 + image18
    axs[1, 2].imshow(image10_18, cmap="gray", alpha=1)
    axs[1, 2].axis('off')
    axs[1, 2].set_title(f"Superposición de 10 y 18")

    # superponer las imagenes 1 y 17 en una misma imagen con un 50% de opacidad
    image1_17 = image1 + image17
    axs[0, 2].imshow(image1_17, cmap="gray", alpha=1)
    axs[0, 2].axis('off')
    axs[0, 2].set_title("Superposición de 1 y 17")
    plt.show()
    
comparar_imagenes_similares()