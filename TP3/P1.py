import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, euclidean_distances
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import svd

def load_data():
    X = pd.read_csv("TP3/dataset02.csv", skiprows=1).to_numpy()
    Y = pd.read_csv("TP3/y.txt").to_numpy().ravel()
    return X, Y

def normalize_dataset(dataset):
    return (dataset - dataset.mean())

def normalize_dataset_martin(dataset):
    return (dataset - dataset.min()) / (dataset.max() - dataset.min())

def compute_covariance_matrix(dataset):
    # Estandarizar los datos
    standardized_data = dataset - np.mean(dataset, axis=0)
    # Calcular la matriz de covarianza
    covariance_matrix = np.cov(standardized_data, rowvar=False)
    return covariance_matrix

def plot_covariance_matrix(cov_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Covarianza')
    plt.show()

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

def calculate_similarity_matrix(dataset, sigma_squared):
    dist_matrix = euclidean_distances(dataset, dataset)
    similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma_squared))
    return similarity_matrix

def plot_similarity_matrix(K, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(K, cmap='viridis')
    plt.title(title)
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

def plot_prediction_errors(errors, dims, method):
    dims_labels = [str(d) for d in dims]
    errors_values = [error for error in errors]
    plt.figure(figsize=(10, 6))
    plt.bar(dims_labels, errors_values, color='skyblue')
    plt.xlabel('Dimensiones Reducidas')
    plt.ylabel('Error de Predicción (MSE)')
    plt.title('Errores de Predicción para Diferentes Dimensiones con ' + method)
    plt.show()

def perform_pca(X_dataset, Y, errors, titles, similarity_matrices, dims, sigma=1):
    pca = PCA(n_components=X_dataset.shape[1])
    pca.fit(X_dataset)
    V = pca.components_.T
    
    for d in dims:
        Vd = V[:, :d]
        Z = np.dot(X_dataset, Vd)
        K_Z = calculate_similarity_matrix(Z, sigma**2)
        similarity_matrices.append(K_Z)
        titles.append(f"Matriz de Similaridad en el Espacio Reducido (d={d}) con PCA")
        
        model = LinearRegression().fit(Z, Y)
        y_pred = model.predict(Z)
        error = mean_squared_error(Y, y_pred)
        errors.append(error)
        
    return similarity_matrices, titles, errors

def perform_svd(X_dataset, Y, errors, titles, similarity_matrices, dims, sigma=1):
    U, S, Vt = np.linalg.svd(X_dataset, full_matrices=False)
    V = Vt.T

    for d in dims:
        Vd = V[:, :d]
        Z = np.dot(X_dataset, Vd)
        K_Z = calculate_similarity_matrix(Z, sigma**2)
        similarity_matrices.append(K_Z)
        titles.append(f"Matriz de Similaridad en el Espacio Reducido (d={d}) con SVD")
        
        model = LinearRegression().fit(Z, Y)
        y_pred = model.predict(Z)
        error = mean_squared_error(Y, y_pred)
        errors.append(error)
        
    return similarity_matrices, titles, errors

def original(X, Y, sigma=1, bool=False):
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

def valores_singulares(dataset):
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)
    # # Graficar los valores singulares en escala semilogarítmica
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(range(1, len(S) + 1), S, 'o-', markersize = 3)
    # plt.xlabel('Índice (i)')
    # plt.ylabel('Valores singulares ($\sigma_i$)')
    # plt.title('Valores singulares $\{\sigma_i\}_{i=1}^{106}$ de $A$ en escala semilogarítmica')
    # plt.grid(True)
    # plt.show()

    # Calcular la proporción acumulada de la suma de los valores singulares
    proporcion_acumulada = np.cumsum(S) / np.sum(S)

    # Graficar la proporción acumulada con una línea más gruesa
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(S) + 1), proporcion_acumulada, 'o-', markersize=3, color="darkcyan", linewidth=2)

    # Agregar una línea desde (0,0) hasta (1, proporcion_acumulada[0]) con la misma grosura
    plt.plot([0, 1], [0, proporcion_acumulada[0]], 'o-', markersize=3, color="darkcyan", linewidth=2)
    
    # Índices y estilos para las líneas en r = 2, 6, 10
    indices = [2, 6, 10]
    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        # Línea vertical hasta el punto de intersección con la proporción acumulada
        plt.plot([idx, idx], [0, proporcion_acumulada[idx-1]], color=color, linestyle=style, linewidth=0.9, label=f'd={idx}')
        # Línea horizontal hasta el punto de intersección con la proporción acumulada
        plt.plot([0, idx], [proporcion_acumulada[idx-1], proporcion_acumulada[idx-1]], color=color, linestyle=style, linewidth=0.9)
    
    # Añadir una línea horizontal en 50% de la varianza acumulada
    umbral_varianza = 0.5
    plt.plot([0, 0.5], [umbral_varianza, umbral_varianza], 'b--', linewidth=1, label='50% de varianza acumulada')

    plt.xlabel('$d$', fontsize=17)
    plt.ylabel(r'$\dfrac{\sum_{i=1}^{d} \sigma_i}{\sum_{i=1}^{n} \sigma_i}$', rotation=0, fontsize=17, labelpad=30, y=0.375)
    # plt.ylabel('$ \sum_{i=1}^{d}} \sigma_i / \sum_{i=1}^{n} \sigma_i$', fontsize=17)
    plt.title('Varianza según dimensión: Proporción de la suma acumulada de $\{\sigma_i\}$ de $A$', fontsize=18)
    plt.legend(fontsize= 14)
    plt.grid(True)
    plt.show()
    
    varianza_d1 = proporcion_acumulada[0]
    varianza_d2 = proporcion_acumulada[1]
    varianza_d3 = proporcion_acumulada[2]
    varianza_d10 = proporcion_acumulada[9]
    diferencia_varianza_10_2 = varianza_d10 - varianza_d2

    print(f"Varianza acumulada para d = 1: {varianza_d1:.4f}")
    print(f"Varianza acumulada para d = 2: {varianza_d2:.4f}")
    print(f"Varianza acumulada para d = 3: {varianza_d3:.4f}")
    print(f"Varianza acumulada para d = 10: {varianza_d10:.4f}")
    print(f"Diferencia de varianza acumulada entre d = 10 y d = 2: {diferencia_varianza_10_2:.4f}")


def plot_singular_values(X, indices=[3, 7, 11]):
    # Cálculo de SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Considerar solo los primeros 102 valores singulares
    S = S[:102]
    
    # Graficar los valores singulares en escala logarítmica
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Valores singulares $\sigma_i$')
    
    # Resaltar los puntos específicos con líneas de diferentes estilos
    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot(idx-1, S[idx-1], 'o', color=color, markersize = 4)  # Poner un punto en el valor singular correspondiente
        plt.vlines(x=idx-1, ymin=S[-1], ymax=S[idx-1], color=color, linestyle=style, label=f'd={idx}')
        plt.hlines(y=S[idx-1], xmin=0, xmax=idx-1, color=color, linestyle=style)
    
    # Configuración del gráfico
    plt.yscale('log')
    plt.xlabel('$i$', fontsize=17)
    plt.ylabel('Valores singulares $\sigma_i$', fontsize=17)
    plt.title('Figura de los valores singulares del dataset $X \{\sigma_i\}_{i=1}^{102}$', fontsize=18)
    plt.legend(fontsize = 14)
    plt.grid(True)
    plt.show()

    
def main():
    X, Y = load_data()
    dims = [2, 6, 10]
    dims_e = [X.shape[1], 2, 6, 10]
    
    K_X, error = original(X, Y)
    similarity_matrices_og = [K_X]
    titles_og = ["Matriz de Similaridad en el Espacio Original"]
    errors_og = [error]
    

    # SVD con matriz original
    # plot_matrices(X)
    # valores_singulares(X)
    # plot_singular_values(X)
    # PCA con matriz normalizada según tutorial
    
    normalized_dataset = normalize_dataset(X)
    plot_singular_values(normalized_dataset)
    
    normalized_dataset = normalize_dataset_martin(X)
    plot_singular_values(normalized_dataset)
    
    # covariance_matrix = compute_covariance_matrix(normalized_dataset)
    
    # plot_covariance_matrix(covariance_matrix)
    
    # plot_matrices(normalized_dataset)
    valores_singulares(normalized_dataset)
    
    
    K_X, error = original(normalized_dataset, Y)
    similarity_matrices_og = [K_X]
    titles_og = ["Matriz de Similaridad en el Espacio Original"]
    errors_og = [error]
    
    similarity_matrices, titles, errors = perform_pca(normalized_dataset, Y, errors_og, titles_og, similarity_matrices_og, dims, 10)
    plot_similarity_matrices(similarity_matrices, titles)
    plot_prediction_errors(errors, dims_e, 'PCA')
    
    # PCA con matriz normalizada según Martín
    normalized_dataset = normalize_dataset_martin(X)
    plot_singular_values(normalized_dataset)
    
    covariance_matrix = compute_covariance_matrix(normalized_dataset)
    
    plot_covariance_matrix(covariance_matrix)
    
    plot_matrices(normalized_dataset)
    valores_singulares(normalized_dataset)

    K_X, error = original(normalized_dataset, Y)
    similarity_matrices_og = [K_X]
    titles_og = ["Matriz de Similaridad en el Espacio Original"]
    errors_og = [error]
    
    similarity_matrices, titles, errors = perform_pca(normalized_dataset, Y, errors_og, titles_og, similarity_matrices_og, dims, 10)
    plot_similarity_matrices(similarity_matrices, titles)
    plot_prediction_errors(errors, dims_e, 'PCA')

if __name__ == "__main__":
    main()
