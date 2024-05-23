import pandas as pd

# Cargar el dataset
dataset = pd.read_csv('TP3/dataset02.csv')
# dataset = pd.read_csv("TP3/dataset02.csv", skiprows=1).to_numpy()

# Normalización de cada columna
normalized_dataset = (dataset - dataset.min()) / (dataset.max())

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Definir la función de similaridad
def gaussian_similarity(x, y, sigma):
    distance = np.linalg.norm(x - y)
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

# Calcular la matriz de similaridad
def similarity_matrix(data, sigma):
    n = data.shape[0]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = gaussian_similarity(data.iloc[i], data.iloc[j], sigma)
    return sim_matrix

# Valores de sigma
sigmas = [0.001, 0.01, 0.1, 1, 10]

# Crear matrices de similaridad para cada sigma y visualizarlas
for sigma in sigmas:
    sim_matrix = similarity_matrix(normalized_dataset, sigma)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis')
    plt.title(f'Matriz de Similaridad (sigma={sigma})')
    plt.show()
    
from sklearn.decomposition import TruncatedSVD

# Aplicar SVD y reducir la dimensionalidad
d_values = [2, 6, 10]
for d in d_values:
    svd = TruncatedSVD(n_components=d)
    reduced_data_svd = svd.fit_transform(normalized_dataset)
    
    # Calcular matriz de similaridad en el espacio reducido
    sim_matrix_svd = similarity_matrix(pd.DataFrame(reduced_data_svd), sigma=1) # Usando sigma=1 como ejemplo
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix_svd, cmap='viridis')
    plt.title(f'SVD: Matriz de Similaridad reducida (d={d}, sigma=1)')
    plt.show()

from sklearn.decomposition import PCA

# Aplicar PCA y reducir la dimensionalidad
for d in d_values:
    pca = PCA(n_components=d)
    reduced_data_pca = pca.fit_transform(normalized_dataset)
    
    # Calcular matriz de similaridad en el espacio reducido
    sim_matrix_pca = similarity_matrix(pd.DataFrame(reduced_data_pca), sigma=1) # Usando sigma=1 como ejemplo
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix_pca, cmap='viridis')
    plt.title(f'PCA: Matriz de Similaridad reducida (d={d}, sigma=1)')
    plt.show()
