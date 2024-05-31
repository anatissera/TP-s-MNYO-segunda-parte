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



# hacer matriz de covarianza para PCA!


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

# SVD
# U, S, Vt = np.linalg.svd(X, full_matrices=False)
# V = Vt.T


# dataset = pd.read_csv('TP3/dataset02.csv')
dataset = pd.read_csv("TP3\dataset02.csv", skiprows=1)

# Normalización de cada columna para PCA
normalized_dataset_martin = (dataset - dataset.min()) / (dataset.max() - dataset.min())
normalized_dataset = dataset - dataset.mean()

# Aplicar SVD
U, S, Vt = np.linalg.svd(normalized_dataset, full_matrices=False)
V = Vt.T


def graficar_matrices(dataset):
    
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)
    V = Vt.T
    
    # Graficar la matriz U
    plt.figure(figsize=(10, 8))
    sns.heatmap(U, cmap='coolwarm')
    plt.title('Matriz U')
    plt.show()

    # # Graficar los valores singulares S
    plt.figure(figsize=(10, 4))
    plt.plot(S, marker='o')
    # plt.yscale('log')
    plt.title('Valores Singulares')
    plt.xlabel('Índice')
    plt.ylabel('Valor Singular')
    plt.show()
    
    # Graficar la matriz V^*
    plt.figure(figsize=(10, 8))
    sns.heatmap(Vt, cmap='coolwarm')
    plt.title('Matriz V*')
    plt.show()


    XV = np.dot(dataset, V)

    # Graficar la matriz XV
    plt.figure(figsize=(12, 8))
    sns.heatmap(XV, cmap='coolwarm', cbar=True)
    plt.title('Matriz $T = AV$')
    plt.xlabel('Componentes')
    plt.ylabel('Muestras')
    plt.show()
    
graficar_matrices(normalized_dataset)
graficar_matrices(normalized_dataset_martin)
graficar_matrices(X)


# Graficar los valores singulares en escala semilogarítmica
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(S) + 1), S, 'o-')
plt.xlabel('Índice (i)')
plt.ylabel('Valores singulares ($\sigma_i$)')
plt.title('Valores singulares $\{\sigma_i\}_{i=1}^{106}$ de $A$ en escala semilogarítmica')
plt.grid(True)
plt.show()

# Calcular la proporción acumulada de la suma de los valores singulares
proporcion_acumulada = np.cumsum(S) / np.sum(S)

# Graficar la proporción acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(S) + 1), proporcion_acumulada, 'o-')
# plt.axhline(y=0.5, color='r', linestyle='--')
# plt.axvline(x=np.argmax(proporcion_acumulada >= 0.5) + 1, color='r', linestyle='--')
# plt.text(np.argmax(proporcion_acumulada >= 0.5) + 1, 0.5, f'r={np.argmax(proporcion_acumulada >= 0.5) + 1}', 
#         color='purple', ha='right')
plt.xlabel('r')
plt.ylabel('$\sum_{i=1}^{r} \sigma_i / \sum_{i=1}^{106} \sigma_i$')
plt.title('Varianza según dimensión: Proporción de la suma acumulada de $\{\sigma_i\}_{i=1}^{106}$ de $A$')
plt.grid(True)
plt.show()



def calculate_similarity_matrix(dataset, sigma_squared):
    dist_matrix = euclidean_distances(dataset, dataset)
    
    similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma_squared))
    
    return similarity_matrix


# Valores de d para reducción de dimensionalidad
d_values = [2, 6, 10, normalized_dataset.shape[1]]

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for ax, d in zip(axes.flatten(), d_values):
    pca = PCA(n_components=d)
    reduced_data_pca = pca.fit_transform(normalized_dataset)
    
    # Calcular matriz de similaridad en el espacio reducido
    sim_matrix_pca = calculate_similarity_matrix(reduced_data_pca, sigma_squared=1)
    
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
    sim_matrix_svd = calculate_similarity_matrix(Z, sigma_squared=1)
    
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
    
sigma = 1.0
def graficar_similaridad(similarity_matrices, titles, errors, separado = False):
    errors = []
    for d in [2, 6, 10, X.shape[1]]:
        Vd = V[:, :d]
        Z = np.dot(X, Vd)

        # similaridad en espacio reducido
        K_Z = calculate_similarity_matrix(Z, sigma**2)
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


def original(X, Y, bool = False):
    # similaridad 
    if bool:
        K_X = calculate_similarity_matrix(np.var(X))
    else:
        K_X = calculate_similarity_matrix(X, sigma)
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



# PCA y similaridades en espacios reducidos
similarities = {}
for d in dims:
    pca = PCA(n_components=d)
    Z_d = pca.fit_transform(X)
    K_Z_d = calculate_similarity_matrix(Z_d, sigma)
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
    plot_similarity_matrix(K_X, "Matriz de Similaridad en el Espacio Original")
    graficar_similaridad(similarity_matrices_og, titles_og, True)
    
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