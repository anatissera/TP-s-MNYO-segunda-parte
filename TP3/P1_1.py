import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

labels = np.loadtxt('tp3/y.txt')

def load_data():
    df = pd.read_csv("TP3/dataset02.csv",  header=None, skiprows=1)

    # # Eliminar la primera columna
    df = df.iloc[:, 1:]
    
    # df.drop(0, axis=1)
    
    # elimina las últimas 6 columnas
    X = df.drop(df.columns[100:], axis=1)
    
    # elimina las primeras 100 columnas
    # X = df.drop(df.columns[:100], axis=1)

    X = df.to_numpy()
    
    # labels = pd.read_csv("TP3/y.txt").to_numpy().ravel()
    labels = np.loadtxt('tp3/y.txt')
    
    return X, labels

def euclidean_distances(X):
    XXT = X @ X.T
    norms = np.diag(XXT)
    distances = np.sqrt(norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * XXT)
    return distances

def normalize_dataset(dataset):
    return (dataset - np.mean(dataset, axis=0))

def normalize_dataset_martin(dataset):
    return (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

def compute_covariance_matrix(dataset):
    covariance_matrix = np.cov(dataset, rowvar=False)
    return covariance_matrix

def plot_covariance_matrix(cov_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Covarianza')
    plt.show()
    
def pca_with_svd(X, d):
    A = normalize_dataset(X)
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    
    print(f"Dimensiones de U: {U.shape}")
    print(f"Dimensiones de S: {S.shape}")
    print(f"Dimensiones de V traspuesta: {Vt.shape}")
    print(f"Dimensiones de V: {V.shape}")
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = Vt[:d, :]
    V_d = V[:, :d]
    
    print(f"Dimensiones de U_{d}: {U_d.shape}")
    print(f"Dimensiones de S_{d}: {S_d.shape}")
    print(f"Dimensiones de V_{d}: {V_d.shape}")
    print(f"Dimensiones de Vt_{d}: {VT_d.shape}")
    
    # Componentes principales
    Z = np.dot(U_d, S_d)
    
    return Z, U_d, S_d, VT_d

def similarity_matrix(X, deviation):
    matrix = normalize_dataset(X)  # Centrar la matriz
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix

def plot_similarity_matrix(matrix, deviation, dim):
    
    sim_matrix = similarity_matrix(matrix, deviation)
    plt.figure()
    # plt.imshow(sim_matrix, cmap='vidris', interpolation='nearest')
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f"Matriz de Similaridad para $d =$ {dim} con desviación {deviation}")
    plt.show()
    
    if dim == 2:
        Z, _, _, _ = pca_with_svd(matrix, dim)
        n_clusters = 2  # Ajustar
        labels = apply_agglomerative_clustering(Z, n_clusters)
        visualize_clusters(Z, labels)
 

def visualize_clusters(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', alpha=0.5)
    plt.title('Distribución con reducción a dos dimensiones y visualización de clusters', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.show()

def apply_agglomerative_clustering(X, n_clusters):
    agglomerative_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative_model.fit_predict(X)
    return labels


def plot_matrices(dataset):
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)
    V = Vt.T
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(U, cmap='coolwarm')
    plt.title('Matriz U')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(S, marker='o')
    plt.title('Valores Singulares')
    plt.xlabel('Índice')
    plt.ylabel('Valor Singular')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(Vt, cmap='coolwarm')
    plt.title('Matriz V*')
    plt.show()

    XV = np.dot(dataset, V)

    plt.figure(figsize=(12, 8))
    sns.heatmap(XV, cmap='coolwarm', cbar=True)
    plt.title('Matriz $T = AV$')
    plt.xlabel('Componentes')
    plt.ylabel('Muestras')
    plt.show()

def plot_similarity_matrices(matrices, titles):
    fig, axs = plt.subplots(2, 2, figsize=(30, 16))
    axs = axs.flatten()
    for ax, K, title in zip(axs, matrices, titles):
        sns.heatmap(K, cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=4) 
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.show()

def compute_reconstruction_error(A, U_d, S_d, VT_d):
    A_d = np.dot(U_d, np.dot(S_d, VT_d))
    errors = np.linalg.norm(A - A_d, axis=1) 
    total_error = np.sum(errors)
    return total_error

def plot_reconstruction_error(X, deviation, dims):
    
    errors = []
    colors = ['darkcyan', 'skyblue', 'steelblue', 'teal']
    
    for i, dim in enumerate(dims):
        Z, U_d, S_d, VT_d = pca_with_svd(X, dim)
        error = compute_reconstruction_error(X, U_d, S_d, VT_d)
        errors.append(error)
        print(f"Error de reconstrucción para d={dim}: {error}")
    
    plt.figure()
    plt.bar(range(len(dims)), errors, tick_label=[str(d) for d in dims], color=colors)
    plt.xlabel('Dimensiones')
    plt.ylabel('Error de reconstrucción (norma 2)')
    plt.title('Error de reconstrucción para las diferentes dimensiones')
    plt.show()
    

def plot_componentes_principales(Z):
    plt.figure(figsize=(12, 8))
    sns.heatmap(Z, cmap='coolwarm', cbar=True)
    plt.title('Matriz $Z = US$')
    plt.xlabel('Componentes')
    plt.ylabel('Muestras')
    plt.show()
    
    
def main():
    X, labels = load_data()
    X = normalize_dataset(X)
    X_reduced2d = pca_with_svd(X, 2)

    
    dims = [2, 6, 10,  X.shape[1]]
    
    deviation = 1
    
    for dim in dims[:-1]:
        Z, U_d, S_d, VT_d = pca_with_svd(X, dim)
        plot_similarity_matrix(Z, deviation, dim)
        plot_componentes_principales(Z)
        
    plot_similarity_matrix(X, deviation, X.shape[1])
    Z, U_d, S_d, VT_d = pca_with_svd(X, X.shape[1])
    plot_componentes_principales(Z)
    
    plot_reconstruction_error(X, deviation, dims)
    
    plot_similarity_matrix(X, deviation, X.shape[1])

if __name__ == '__main__':
    main()