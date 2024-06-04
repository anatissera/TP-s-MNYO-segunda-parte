import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

def load_data():
    df = pd.read_csv("TP3/dataset02.csv", header=None, skiprows=1)
    df = df.iloc[:, 1:]
    X = df.to_numpy()
    labels = np.loadtxt('tp3/y.txt')
    return X, labels

def normalize_dataset(dataset):
    return (dataset - np.mean(dataset, axis=0))

def pca_with_svd(X, d):
    A = normalize_dataset(X)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Z = np.dot(U[:, :d], np.diag(S[:d]))
    return Z, U[:, :d], np.diag(S[:d]), Vt[:d, :]

def similarity_matrix(X, deviation):
    matrix = normalize_dataset(X)
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix

def plot_similarity_matrix(matrix, deviation, dim):
    sim_matrix = similarity_matrix(matrix, deviation)
    plt.figure()
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f"Matriz de Similaridad para $d =$ {dim} con desviación {deviation}")
    plt.show()

def evaluate_clustering(X, labels):
    mask = labels != -1
    silhouette = silhouette_score(X[mask], labels[mask])
    davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
    return silhouette, davies_bouldin

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

def plot_dendrogram(X, method='ward'):
    linked = linkage(X, method=method)
    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='lastp', p=12)
    plt.title('Dendrograma')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Distancia')
    plt.show()

def main():
    X, labels = load_data()
    X = normalize_dataset(X)
    
    dims = [2, 6, 10, X.shape[1]]
    
    for dim in dims:
        X_reduced, _, _, _ = pca_with_svd(X, dim)
        if dim == 2:
            # Determinar el número óptimo de clusters (ej. usar el dendrograma)
            plot_dendrogram(X_reduced)
            
            # Aplicar Agglomerative Clustering
            n_clusters = 2  # Ajustar según el dendrograma
            labels = apply_agglomerative_clustering(X_reduced, n_clusters)
            silhouette, db = evaluate_clustering(X_reduced, labels)
            print(f"Agglomerative Clustering para d={dim}:")
            print(f'Silhouette Score: {silhouette}')
            print(f'Davies-Bouldin Score: {db}')
            visualize_clusters(X_reduced, labels)

if __name__ == '__main__':
    main()
