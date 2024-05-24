import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Análisis de Componentes Principales
#  identifica los componentes principales de los datos -> los vectores ortogonales 
# no correlacionados entre sí y ordenados jerárquicamente que maximizan la varianza 𝜎^2 de las mediciones
# emplea SVD para calcular los componentes principales
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, euclidean_distances
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd


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
        
        



# X = pd.read_csv('dataset.csv').values
# y = pd.read_csv('y.txt').values

X = pd.read_csv("TP3/dataset02.csv", skiprows=1).to_numpy()
Y = pd.read_csv("TP3/y.txt").to_numpy().ravel()
dims = [2, 6, 10, X.shape[1]]
# dims = [2, 6, 10]

# SVD y PCA
U, S, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


# dataset = pd.read_csv('TP3/dataset02.csv')
dataset = pd.read_csv("TP3\dataset02.csv", skiprows=1)

# Normalización de cada columna
normalized_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())

# Aplicar SVD
U, S, Vt = np.linalg.svd(normalized_dataset, full_matrices=False)
V = Vt.T


# Graficar la matriz U
plt.figure(figsize=(10, 8))
sns.heatmap(U, cmap='coolwarm')
plt.title('Matriz U')
plt.show()

# # Graficar los valores singulares S
# plt.figure(figsize=(10, 4))
# plt.plot(S, marker='o')
# # plt.yscale('log')
# plt.title('Valores Singulares')
# plt.xlabel('Índice')
# plt.ylabel('Valor Singular')
# plt.show()

from brokenaxes import brokenaxes

# plt.figure(figsize=(10, 4))
# plt.semilogy(S, marker='o')
# plt.title('Valores Singulares (Escala Semi-Logarítmica)')
# plt.xlabel('Índice (i)')
# plt.ylabel('Valor Singular ($\sigma_i$)')

# plt.show()

fig = plt.figure(figsize=(10, 4))
bax = brokenaxes(ylims=((1.8302e-15, 2.33357e-13), (1.8, 440)), hspace=0.1)
bax.semilogy(S, marker='o')
bax.set_title('Valores Singulares')
bax.set_xlabel('Índice (i)')
bax.set_ylabel('Valor Singular ($\sigma_i$)')
plt.show()

# plt.figure(figsize=(10, 4))
# plt.semilogy(S, marker='o')
# plt.title('Valores Singulares (Escala Semi-Logarítmica)')
# plt.xlabel('Índice (i)')
# plt.ylabel('Valor Singular ($\sigma_i$)')

# plt.show()

# from brokenaxes import brokenaxes

# # Graficar los valores singulares S con ejes interrumpidos
# fig = plt.figure(figsize=(10, 4))
# bax = brokenaxes(ylims=((0, 45), (230, 250)), hspace=0.1)
# bax.plot(range(len(S)), S, marker='o')
# bax.set_title('Valores Singulares')
# bax.set_xlabel('Índice')
# bax.set_ylabel('Valor Singular')
# plt.show()

# Graficar la matriz V^*
plt.figure(figsize=(10, 8))
sns.heatmap(Vt, cmap='coolwarm')
plt.title('Matriz V*')
plt.show()


from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

# Valores de d para reducción de dimensionalidad
d_values = [2, 6, 10, normalized_dataset.shape[1]]

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for ax, d in zip(axes.flatten(), d_values):
    pca = PCA(n_components=d)
    reduced_data_pca = pca.fit_transform(normalized_dataset)
    
    # Calcular matriz de similaridad en el espacio reducido
    sim_matrix_pca = np.exp(-euclidean_distances(reduced_data_pca, reduced_data_pca)**2 / (2 * np.var(reduced_data_pca)))
    
    # Graficar la matriz de similaridad
    sns.heatmap(sim_matrix_pca, cmap='viridis', ax=ax)
    ax.set_title(f'PCA: Matriz de Similaridad reducida (d={d})')
    ax.tick_params(axis='both', which='major', labelsize=4)

# plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for ax, d in zip(axes.flatten(), d_values):
    Z = np.dot(U[:, :d], np.diag(S[:d]))
    
    # Calcular matriz de similaridad en el espacio reducido
    sim_matrix_svd = np.exp(-euclidean_distances(Z, Z)**2 / (2 * np.var(Z)))
    
    # Graficar la matriz de similaridad
    sns.heatmap(sim_matrix_svd, cmap='viridis', ax=ax)
    ax.set_title(f'SVD: Matriz de Similaridad reducida (d={d})')
    ax.tick_params(axis='both', which='major', labelsize=4)

# plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()


# Matrices de similaridad separadas
def plot_similarity_matrix(K, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(K, cmap='viridis')
    plt.title(title)
    plt.show()
    

def graficar_similaridad(similarity_matrices, titles, errors, separado = False):
    errors = []
    for d in [2, 6, 10, X.shape[1]]:
        Vd = V[:, :d]
        Z = np.dot(X, Vd)

        # similaridad en espacio reducido
        K_Z = np.exp(-euclidean_distances(Z, Z)**2 / (2 * np.var(Z)))
        if separado:
            plot_similarity_matrix(K_Z, f"Matriz de Similaridad en el Espacio Reducido (d={d})")
     
        similarity_matrices.append(K_Z)
        titles.append(f"Matriz de Similaridad en el Espacio Reducido (d={d})")
        
        # regresión lineal en el espacio reducido
        # pca = PCA(n_components=d)
        # Z_d = pca.fit_transform(X)
        model = LinearRegression().fit(Z, Y)
        y_pred = model.predict(Z)
        error = mean_squared_error(Y, y_pred)
        errors.append((d, error))
        # print(f"Dimensión reducida a {d}, error de predicción: {error}")
        
    if not separado:
        return similarity_matrices, titles, errors


# en subplots
def plot_similarity_matrices(matrices, titles):
    fig, axs = plt.subplots(2, 2, figsize=(30, 16))
    axs = axs.flatten()
    for ax, K, title in zip(axs, matrices, titles):
        sns.heatmap(K, cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=4) 
        
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.show()


def original(X, Y, bool = True):
    # similaridad 
    if bool:
        K_X = np.exp(-euclidean_distances(X, X)**2 / (2 * np.var(X)))
    else:
        K_X = calculate_similarity(X, sigma)
    # regresión lineal
    model = LinearRegression().fit(X, Y)
    y_pred = model.predict(X)
    error = mean_squared_error(Y, y_pred)
    
    return K_X, error


   
# Visualización de los componentes principales
def componentes_principales(dims, X):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    for i, d in enumerate(dims[:-1]):
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
def errores_prediccion(errors, dims):
    dims_labels = [str(d) for d in dims]
    errors_values = [error for _, error in errors]

    plt.figure(figsize=(10, 6))
    plt.bar(dims_labels, errors_values, color='skyblue')
    plt.xlabel('Dimensiones Reducidas')
    plt.ylabel('Error de Predicción (MSE)')
    plt.title('Errores de Predicción para Diferentes Dimensiones')
    plt.show()



# otra manera con la fórmula de similaridad del enunciado (?
# y con     pca = PCA(n_components=d)
#           Z_d = pca.fit_transform(X)

def calculate_similarity(X, sigma):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return K

sigma = 1.0     # ajustar 


# PCA y similaridades en espacios reducidos
similarities = {}
for d in dims:
    pca = PCA(n_components=d)
    Z_d = pca.fit_transform(X)
    K_Z_d = calculate_similarity(Z_d, sigma)
    similarities[d] = K_Z_d

# Error de predicción
errors_2 = []
for d in dims:
    pca = PCA(n_components=d)
    Z_d = pca.fit_transform(X)
    model = LinearRegression()
    model.fit(Z_d, Y)
    Y_pred = model.predict(Z_d)
    error = mean_squared_error(Y, Y_pred)
    errors_2.append((d, error))


def similarity_matrix_conheatmap(K_X, similarities):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.heatmap(K_X, ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('Similaridad en el Espacio Original')

    for i, d in enumerate(dims[:-1]):
        row = (i + 1) // 2
        col = (i + 1) % 2
        sns.heatmap(similarities[d], ax=axes[row, col], cmap='viridis')
        axes[row, col].set_title(f'Similaridad en el Espacio Reducido d={d}')

    plt.subplots_adjust(hspace=1.5, wspace=0.5)
    plt.tight_layout()
    plt.show()


def main():
    
    K_X, error = original(X, Y)
    similarity_matrices_og = [K_X]
    titles_og = ["Matriz de Similaridad en el Espacio Original"]
    errors_og = [('Original', error)]

    similarity_matrices, titles, errors = graficar_similaridad(similarity_matrices_og, titles_og, errors_og)

    # similaridad
    #varios subplots
    plot_similarity_matrices(similarity_matrices, titles)

    #separado
    # plot_similarity_matrix(K_X, "Matriz de Similaridad en el Espacio Original")
    # graficar_similaridad(similarity_matrices_og, titles_og, True)
    
    errores_prediccion(errors, dims)
    errores_prediccion(errors_2, dims) # ?
    componentes_principales(dims, X)
    
    
    # con otro método
    # K_X, error = original(X, Y, False)
    # similarity_matrices_og = [K_X]
    # errors_og = [('Original', error)]

    # similarity_matrix_conheatmap(K_X, similarities)

if __name__ == "__main__":
    main()