import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import euclidean_distances
from sklearn.cluster import AgglomerativeClustering

labels = np.loadtxt('tp3/y.txt')
labels = labels - np.mean(labels)

def processMatrix(matrix):
    matrix = pd.read_csv(path, header=None)
    # Elimino la primera fila y columna que son los nombres de las columnas y filas
    matrix = matrix.drop(0, axis=0)
    matrix = matrix.drop(0, axis=1)
    
   
    # Elimino las primeras 100 columnas
    # matrix = matrix.drop(matrix.columns[:100], axis=1)

    # elegir las columnas entre un n y otro
    # matrix = matrix.iloc[:, 104:106]
    
    # elimina las ultimas 6 columnas
    # matrix = matrix.drop(matrix.columns[100:], axis=1)


    # Convierto la matriz a un array de numpy
    matrix = matrix.to_numpy()
    matrix = matrix - np.mean(matrix, axis=0) # esta cosa centra la matriz

    return matrix

def euclidean_distances(X):
    # Calculamos la matriz de productos internos
    XXT = X @ X.T
    # Extraemos la diagonal (normas cuadradas de cada punto)
    norms = np.diag(XXT)
    # Calculamos la matriz de distancias euclidianas usando broadcasting
    distances = np.sqrt(norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * XXT)
    return distances

def similarity_matrix(matrix, deviation):
    # diff = matrix[:, None] - matrix
    # dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    # sim_matrix = np.exp(-np.sqrt(dist_sq) / (2 * deviation**2))
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix


def show_similarity_matrix(matrix, deviation):

    sim_matrix = similarity_matrix(matrix, deviation)

    plt.figure()
    plt.imshow(sim_matrix)
    plt.colorbar()
    plt.title(f"Matriz de similaritud con desviación {deviation}")
    plt.show()
    
def reduceDimensions(matrix, deviation, dimension):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    print(f"Dimensiones de U: {u.shape}")
    print(f"Dimensiones de S: {s.shape}")
    print(f"Dimensiones de Vt: {vt.shape}")
    print()    

    # Reducimos la dimensión de la matriz
    U_reduced = u[:, :dimension]
    S_reduced = np.diag(s[:dimension])
    matrix_svd = U_reduced @ S_reduced
    print(f"Dimensiones de MatrizSVD: {matrix_svd.shape}")
    
    # sim_matrix = similarity_matrix(matrix_svd, deviation)
    # sim_matrix_d10 = similarity_matrix(matrix_svd, 10)
    # print(f"Dimensiones de MatrizSimilitudSVD: {sim_matrix.shape}")
    
    # plt.figure()
    # plt.imshow(sim_matrix)
    # plt.colorbar()
    # plt.title(f"Matriz de similaritud con desviación {deviation} y reducción de dimensiones a {dimension}")
    # plt.show()
    
    # plt.figure()
    # plt.imshow(sim_matrix_d10)
    # plt.colorbar()
    # plt.title(f"Matriz de similaritud con desviación 10 y reducción de dimensiones a {dimension}")
    # plt.show()
    
    if dimension == 2:
        plt.figure()
        plt.bar(range(len(s)), s, color='darkcyan')
        plt.title("Valores singulares", fontsize = 14)
        # plt.xticks(range(len(vt[0])), [str(i+100) for i in range(len(vt[0]))])
        plt.xlabel("$i$", fontsize = 16)
        plt.ylabel("valor singular", fontsize = 18)
        plt.show()
        
        plt.figure()
        plt.bar(range(len(vt[0])), (vt[0, :]), color='darkcyan')
        plt.title("Primer Vector de $V^T$", fontsize = 16)
        plt.xlabel("Componente", fontsize = 14)
        plt.ylabel("Valor del Componente", fontsize = 14)
        # plt.xticks(range(len(vt[0])), [str(i+101) for i in range(len(vt[0]))])
        plt.grid()
        plt.show()
        
        n_clusters = 2  # Ajustar según el dendrograma
        labels2 = apply_agglomerative_clustering(matrix_svd, n_clusters)
        visualize_clusters(matrix_svd, labels2)
        
        return
    
def visualize_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', alpha=0.5)
    plt.title('Distribución de las componentes 100 y 101', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.show()

def apply_agglomerative_clustering(X, n_clusters):
    agglomerative_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative_model.fit_predict(X)
    return labels
    
def iterateDimensions(matrix, deviation, dimensions):
        dimension = dimensions[0] 
        for i in range(1,5):
            print(f"Desviación: {deviation}")
            reduceDimensions(matrix, deviation, dimension)
            dimension = dimensions[i]  
    
def iterateDeviation(matrix):
    deviation = 10
    for i in range(5):
        print(f"Desviación: {deviation}")
        show_similarity_matrix(matrix, deviation)
        deviation = deviation/10

def plot_componentes_principales(Z):
    plt.figure(figsize=(12, 8))
    sns.heatmap(Z, cmap='coolwarm', cbar=True)
    plt.title('Matriz $Z = US$')
    plt.xlabel('Componentes')
    plt.ylabel('Muestras')
    plt.show()
    
def similarity_matrix(matrix, deviation):
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix

def show_similarity_matrix(matrix, deviation):
    sim_matrix = similarity_matrix(matrix, deviation)
    plt.figure()
    plt.imshow(sim_matrix)
    plt.colorbar()
    plt.title(f"Matriz de similaridad con desviación {deviation}")
    plt.show()
    
def plot_singular_values(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(S)), S, color='darkcyan')
    plt.yscale('log')
    plt.xlabel('$i$', fontsize=17)
    plt.ylabel('Valores singulares $\sigma_i$', fontsize=17)
    plt.title('Figura de los valores singulares del dataset $X \{\sigma_i\}_{i=1}^{102}$', fontsize=18)
    plt.grid(False)
    plt.show()
    
def plot_features(X):
    plt.figure(figsize=(10, 8))
    sns.heatmap(X, cmap='coolwarm', cbar=True)
    plt.title('Matriz de datos')
    plt.show()
    
def plot_features_bar(X):
    plt.figure(figsize=(10, 8))
    plt.bar(range(X.shape[1]), np.mean(X, axis=0), color='darkcyan')

# Ruta del archivo de datos
path = 'TP3/dataset02.csv'
# Procesar la matriz
matrix = processMatrix(path)



# # Mostrar la matriz de similaridad
# deviation = 1
# # plot_singular_values(matrix)
# show_similarity_matrix(matrix, deviation)

# deviation = 1
# dimensions = [2, 3, 6, 10, 102]

# # iterateDeviation(matrix)
# iterateDimensions(matrix, deviation, dimensions)


plt.figure(figsize=(8, 8))  # Ajusta estos valores según tus necesidades
plt.imshow(matrix, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Matriz original", fontsize=18)
# plt.xticks(range(0, 106, 5))
plt.xlabel("Características", fontsize=15)
plt.ylabel("Muestras", fontsize=15)
plt.tight_layout()
# hace una linea trazada en el eje X = 102
if matrix.shape[1] >= 100:
    plt.axvline(x=100, color='r')
plt.show()

correlation_matrix= similarity_matrix(matrix.T, 1)

# Plotear el heatmap de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='Spectral', center=0)
plt.title('Similaridad entre columnas de la matriz de datos', fontsize=16)
plt.show()
