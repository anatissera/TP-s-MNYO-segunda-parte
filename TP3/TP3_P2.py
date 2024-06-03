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

# 

def plot_singular_values(S):
    # Valores singulares {𝜎𝑖}19
    # 𝑖=1 de 𝐴 en escala semilogarítmica
    # respecto al eje de ordenadas. Se observa que se hallan ordenados
    # de forma decreciente y que el primer valor singular 𝜎1 duplica
    # al siguiente valor singular 𝜎2, frente a un menor decrecimiento
    # valor a valor para 𝑖 subsiguientes.

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
    axs.plot(cumsum, marker="o",markersize=5, color="slateblue", label="Proporción de la suma acumulada")

# graficar unas lineas que marquen el punto de 72,5% de la suma
# for idx, style, color in zip(indices, styles, colors):
 # plt.plot(idx-1, S[idx-1], 'o', color=color, markersize = 4) 
    # plt.vlines(x=idx-1, ymin=S[-1], ymax=S[idx-1], color=color, linestyle=style, label=f'd={(idx - 1)}')
 # plt.hlines(y=S[idx-1], xmin=0, xmax=idx-1, color=color, linestyle=style)
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
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for d, ax in zip(d_values, axs.flatten()):
        Z = np.dot(X, Vt[:d].T)
        X_reconstructed = np.dot(Z, Vt[:d])
        images_reconstructed = X_reconstructed.reshape(n, p, p)
        ax.imshow(images_reconstructed[11], cmap="gray")
        # mostrar el porcentaje de error en cada imagen
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
        # mostrar el porcentaje de error en cada imagen
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

    ax.bar(np.arange(n), error1, label=f"d={d1}", color="slateblue")
    ax.bar(np.arange(n), error2, label=f"d={d2}", color="salmon")
    ax.set_xticks(np.arange(n))
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

    # Utilizando esta ultima representación aprendida con el dataset 2 ¿Qué error de reconstrucción obtienen si utilizan la misma compresión (con la misma base de d dimensiones obtenida del dataset 2) para las imagenes dataset_imagenes1.zip?

    # usar la representacion aprendida con el dataset 2
    # usar esta compresion para las imagenes del dataset 1
    #cargar imagenes del dataset1
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
    ax.bar(np.arange(n), errors, color="salmon")
    ax.set_title("Error de reconstrucción")
    plt.show()

    # graficar las imagenes reconstruidas de una imagen del dataset 1
    fig, axs = plt.subplots(1, 2)
    Z = np.dot(X[1], Vt[:d].T)
    X_reconstructed = np.dot(Z, Vt[:d])
    images_reconstructed = X_reconstructed.reshape(p, p)
    axs[0].imshow(images[1], cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(images_reconstructed, cmap="gray")
    axs[1].set_title("Reconstruida")
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
    fig, axs = plt.subplots(2, 2)
    for d, ax in zip(d_values, axs.flatten()):
        similarity_matrixe = similarity_matrix(U, S, d)
        ax.imshow(similarity_matrixe, cmap="viridis")
        ax.set_title(f"d={d}")
    plt.show()

d_values = [2, 6, 12, 19]
plot_similarity_matrix(U, S, d_values)

def comparar_imagenes_similares():
    # sabemos que la imagen 1 es similar a a la 17, y la 10 con la 18, hacer un subplot de las 4 imagenes y mostrar la similitud entre ellas
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

    similarity_matrixe = similarity_matrix(U, S, 19)
    print(similarity_matrixe[1, 17])
    print(similarity_matrixe[10, 18])

    similarity_matrixe = similarity_matrix(U, S, 2)
    print(similarity_matrixe[10, 18])

    #superponer las imagenes 10 y 18 en una misma imagen con un 50% de opacidad
    image10_18 = image10 + image18
    axs[1, 2].imshow(image10_18, cmap="gray", alpha=1)
    axs[1, 2].axis('off')
    axs[1, 2].set_title(f"Superposición de 10 y 18")

    # calcular porcentaje de similaridad
    

    # superponer las imagenes 1 y 17 en una misma imagen con un 50% de opacidad
    image1_17 = image1 + image17
    axs[0, 2].imshow(image1_17, cmap="gray", alpha=1)
    axs[0, 2].axis('off')
    axs[0, 2].set_title("Superposición de 1 y 17")
    plt.show()
    
comparar_imagenes_similares()



# En latex tengo que escribir un informe que se centra mas que nada en SVD y reducción de la dimensionalidad. El punto 1 ya esta todo hecho. Este archivo tiene que ver con el punto 2.

# En el punto 1 en latex, escribimos una especie de introduccion al problema asi:
# En esta seccion, se llevar ́a a cabo un an ́alisis detallado del dataset X utilizando la descomposici ́on en
# valores singulares (SVD) y el an ́alisis de componentes principales (PCA). El dataset X est ́a compuesto
# por n muestras xi ∈ Rp, donde X es una matriz de dimensiones n × p. Dado que el conjunto de datos
# presenta una alta dimensionalidad, es crucial entender c ́omo se distribuyen las muestras visualmente. Se
# asume que las muestras no se distribuyen uniformemente en el espacio Rp, permitiendo la identificaci ́on
# de grupos de muestras similares (clusters) con alta similitud entre s ́ı.
# Primero, se determinar ́an las componentes principales de X y se definir ́a una m ́etrica de similitud
# entre muestras. Posteriormente, se reducir ́a la dimensionalidad del dataset a diferentes valores de d para
# evaluar cu ́al de estas dimensiones es m ́as adecuada para el an ́alisis. Se estudiar ́a c ́omo se relacionan estos
# resultados con los valores singulares de X y se extraer ́an conclusiones al respecto.
# Adicionalmente, se evaluar ́a la representatividad de las dimensiones originales del dataset en relaci ́on
# con las dimensiones obtenidas mediante SVD, identificando cu ́ales de las dimensiones originales son m ́as
# importantes.
# Este an ́alisis se complementar ́a con la resoluci ́on de un problema de cuadrados m ́ınimos, en el que se
# buscar ́a encontrar el vector β que minimice la norma ∥X ˆβ − y∥2. Esto permitir ́a predecir las etiquetas y
# minimizando el error de predicci ́on y evaluar c ́omo la reducci ́on de dimensionalidad a diferentes valores de
# d impacta en la precisi ́on de las predicciones, identificando cu ́al de estas mejora dicha predicci ́on.
# En resumen, este estudio tiene como objetivo evaluar la eficacia de la reducci ́on de dimensionalidad
# para la identificaci ́on de clusters y la mejora en la predicci ́on de etiquetas, proporcionando as ́ı una
# caracterizaci ́on tanto cualitativa como cuantitativa del dataset analizado. Adem ́as, se discutir ́a c ́omo el
# ruido y la redundancia en los datos se ven afectados al analizar componentes principales de diferentes
# dimensionalidades.

"""
En esta sección, se estudiará la compresión de imágenes utilizando la descomposición en valores singulares (SVD). La compresión de imágenes es un proceso que permite reducir la cantidad de información, permitiendo almacenar y transmitir imágenes de forma más eficiente. Esto nos ayudará a entender cómo se relaciona la reducción de dimensionalidad con la calidad de las imágenes reconstruidas, identificando cuál es el número mínimo de dimensiones para obtener una compresión efectiva.

\subsection {Descomposición en valores singulares}

En este caso, usaremos primero el \textit{dataset imágenes 1} donde se considerarán 19 imágenes de 28 × 28 píxeles, que se representan como vectores $X_i \in \mathbb{R}^{28 \times 28}$, permitiendo la construcción de una matriz de datos $A \in \mathbb{R}^{19 \times 784}$, donde los vectores $X_i$ representan las imágenes en cada fila de la matriz. De esta manera, podemos aplicar SVD a la matriz $A$:

\begin{equation}
    A = U \Sigma V^T
\end{equation}

donde $U \in \mathbb{R}^{19 \times 19}$ y $V \in \mathbb{R}^{784 \times 784}$ son matrices ortogonales cuyas columnas son autovectores de $AA^T$ y $A^TA$ respectivamente, y $\Sigma \in \mathbb{R}^{19 \times 784}$ es una matriz diagonal con los valores singulares de $A$.

A partir de esta matriz, somos capaces de aprender una representación de baja dimensión de las imágenes mediante la descomposición en valores singulares, estos valores singulares nos permitirán identificar cuáles son las componentes más importantes de la descomposición y cómo se relacionan con la calidad de las imágenes reconstruidas. Esto se puede ver en los siguientes gráficos:

% subfigure
\begin{figure}[H]    
\centering
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{Figures/Gráficos/Punto 2/Singular_Values.png}
        \caption{Valores singulares $\{\sigma_i\}_{i=0}^{18}$ de $A$ en escala semilogarítmica, donde se observa que estan ordenados de forma decreciente y los primeros valores singulares aportan más información que los últimos.}
    \label{fig: Singular_Values}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{Figures/Gráficos/Punto 2/Suma_Acumulada.png}
    \caption{Suma acumulada de los valores singulares  $\{\sigma_i\}_{i=0}^{18}$ respecto de la desviación total. Mostrando particularmente que tomando la primera mitad de los valores singulares, se consigue cerca de un 72,5\% de la información de las imágenes originales.}
    \label{fig: Suma_Acumulada}
\end{subfigure}
\caption{Analisis de los valores singulares de $A$.}
\label{fig: Singular_Values_info}
\end{figure}

\quad Como podemos observar en la Figura \ref{fig: Singular_Values}, los valores singulares $\{\sigma_i\}_{i=0}^{18}$ representan el aporte de información de los componentes de la descomposición SVD, donde se observa que de manera decreciente los primeros valores singulares aportan más información que los últimos, esto quiere decir, que a la hora de reconstruir imagenes, tenemos que tener muy en cuenta los primeros valores singulares, ya que tendrán un mayor impacto en la calidad de la imagen reconstruida. \\
\quad Esto se respalda en la Figura \ref{fig: Suma_Acumulada} donde se puede ver que solo usando la primera mitad de los valores singulares, se puede reconstruir aproximadamente el $75\%$ de la información de las imágenes originales, lo que nos indica que la reducción de dimensionalidad tomando valores singulares de la primera mitad, no implica una pérdida significativa de información, caso contrario a tomar los valores singulares de la segunda mitad, donde la pérdida de información es mucho mayor. Es decir, que en una reconstrucción de imágenes, no importa tanto la cantidad de valores singulares que se tomen, sino la calidad de los mismos.\\

\subsection {Reconstrucción de imágenes}

\quad Habiendo sacado nuestras primeras conclusiones sobre la importancia de los valores singulares, procedemos a reconstruir las imágenes originales a partir de la descomposición SVD, visualizando en forma matricial $28 \times 28$ las imágenes reconstruidas luego de compresión con distintos valores de $d$ dimensiones.\\

# explicar el proceso para comprimir las imagenes y reconstruirlas
#  U_reduced = U[:, :d]
#  S_reduced = np.diag(S[:d])
#  Z = np.dot(U_reduced, S_reduced)

En orden de comprimir las imagenes usando la descomposción SVD, se toman los primeros $d$ valores singulares de la matriz $A$, y se calcula la matriz $Z$ que representa la matriz de datos comprimida.

\begin{equation}
    U_{reduced} = U[:, :d]
    S_{reduced} = np.diag(S[:d])
    Z = U_{reduced} \Sigma_{reduced}
\end{equation}

Luego, para reconstruir las imágenes, se calcula la matriz de datos reconstruida $A_{reconstructed}$ a partir de la matriz $Z$ y los autovectores $V$ de la descomposición SVD.

\begin{equation}
    A_{reconstructed} = Z V^T
\end{equation}

Finalmente, se visualizan las imágenes reconstruidas en forma matricial $28 \times 28$ como vemos a continuación:

"""