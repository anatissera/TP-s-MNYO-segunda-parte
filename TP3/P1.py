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
    # X = pd.read_csv("TP3/dataset02.csv", skiprows=1, usecols=range(1, X.shape[1]+1)).to_numpy()
    df = pd.read_csv("TP3/dataset02.csv", skiprows=1)

    # # Eliminar la primera columna
    # df = df.iloc[:, 1:]
    df.drop(0, axis=1)

    # Convertir el DataFrame en un array de NumPy
    X = df.to_numpy()
    
    label = pd.read_csv("TP3/y.txt").to_numpy().ravel()
    
    return X, label

def normalize_dataset(dataset):
    return (dataset - np.mean(dataset, axis=0))

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
    
    print(f"Dimensiones de $U_{d}$: {U_d.shape}")
    print(f"Dimensiones de $S_{d}$: {S_d.shape}")
    print(f"Dimensiones de $V_{d}$: {V_d.shape}")
    print(f"Dimensiones de $V^T_{d}$: {VT_d.shape}")
    
    # Componentes principales
    Z = np.dot(U_d, S_d)
    
    
    return Z, U_d, S_d, VT_d

def similarity_matrix(X, deviation):
    matrix = normalize_dataset(X)  # Centrar la matriz
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix

# def calculate_similarity_matrix(dataset, sigma_squared):
#     dist_matrix = euclidean_distances(dataset, dataset)
#     similarity_matrix = np.exp(-dist_matrix**2 / (2 * sigma_squared))
#     return similarity_matrix

def plot_similarity_matrix(matrix, deviation):
    
    sim_matrix = similarity_matrix(matrix, deviation)
    plt.figure()
    plt.imshow(sim_matrix, cmap='hot', interpolation='nearest')
    # plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f"Matriz de Similaridad con desviación {deviation}")
    plt.show()

    
def show_similarity_matrix(matrix, deviation):
    sim_matrix = similarity_matrix(matrix, deviation)
    plt.figure()
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f"Matriz de Similaridad con desviación {deviation}")
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
        K_Z = similarity_matrix(Z, sigma**2)
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
        K_Z = similarity_matrix(Z, sigma**2)
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
        K_X = similarity_matrix(np.var(X))
    else:
        K_X = similarity_matrix(X, sigma)
    # regresión lineal
    model = LinearRegression().fit(X, Y)
    y_pred = model.predict(X)
    error = mean_squared_error(Y, y_pred)
    
    return K_X, error

def valores_singulares_acumulada(dataset):
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)

    # proporcion_acumulada = np.cumsum(S)
    # proporcion_acumulada /= proporcion_acumulada[-1]
    
    # proporcion_acumulada = np.cumsum(S) / np.arange(1, len(S) + 1)
    # proporcion_acumulada = np.cumsum(S) / np.sum(S)
    
    S_squared = S**2
    proporcion_acumulada = np.cumsum(S_squared) / np.sum(S_squared)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(S) + 1), proporcion_acumulada, 'o-', markersize=3, color="darkcyan", linewidth=2)

    plt.plot([0, 1], [0, proporcion_acumulada[0]], 'o-', markersize=3, color="darkcyan", linewidth=2)
   
    indices = [2, 6, 10]
    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot([idx, idx], [0, proporcion_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5, label=f'd={idx}')
        plt.plot([0, idx], [proporcion_acumulada[idx-1], proporcion_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5)

    umbral_varianza = 0.5
    plt.plot([0, len(S)], [umbral_varianza, umbral_varianza], 'b--', linewidth=1.4, label='50% de varianza acumulada')

    plt.xlabel('$d$', fontsize=17)
    plt.ylabel(r'$\dfrac{\sum_{i=1}^{d} \sigma_i ^2}{\sum_{i=1}^{n} \sigma_i ^2}$', rotation=0, fontsize=17, labelpad=30, y=0.375)
    plt.title('Varianza según dimensión: Proporción de la suma acumulada de $\{\sigma_i\}$ de $A$', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()
    
    varianza_d1 = proporcion_acumulada[0]
    varianza_d2 = proporcion_acumulada[1]
    varianza_d6 = proporcion_acumulada[5]
    varianza_d10 = proporcion_acumulada[9]
    diferencia_varianza_10_2 = varianza_d10 - varianza_d2
    diferencia_varianza_6_2 = varianza_d6 - varianza_d2
    diferencia_varianza_10_6 = varianza_d10 - varianza_d6

    print(f"Varianza acumulada para d = 1: {varianza_d1:.4f}")
    print(f"Varianza acumulada para d = 2: {varianza_d2:.4f}")
    print(f"Varianza acumulada para d = 6: {varianza_d6:.4f}")
    print(f"Varianza acumulada para d = 10: {varianza_d10:.4f}")
    print(f"Diferencia de varianza acumulada entre d = 10 y d = 2: {diferencia_varianza_10_2:.4f}")
    print(f"Diferencia de varianza acumulada entre d = 6 y d = 2: {diferencia_varianza_6_2:.4f}")
    print(f"Diferencia de varianza acumulada entre d = 10 y d = 6: {diferencia_varianza_10_6:.4f}")

def media_acumulada_valores_singulares(dataset):
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)

    diferencias = np.diff(S)
    
    acumulacion_diferencias = np.cumsum(diferencias)
    
    media_acumulada = np.cumsum(diferencias) / np.arange(1, len(diferencias) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(media_acumulada) + 1), media_acumulada, 'o-', markersize=3, color="darkcyan", linewidth=2)
    
    indices = [2, 6, 10]
    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot([idx, idx], [0, media_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5, label=f'd={idx}')
        plt.plot([0, idx], [media_acumulada[idx-1], media_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5)
  
    umbral_varianza = 0.5
    plt.plot([0, len(S)], [umbral_varianza, umbral_varianza], 'b--', linewidth=1.4, label='50% de varianza acumulada')


    plt.xlabel('$i$', fontsize=17)
    plt.ylabel('Media Acumulada de las Diferencias de los Valores Singulares', fontsize=17)
    plt.title('Media Acumulada de las Diferencias de los Valores Singulares de $A$', fontsize=18)
    plt.grid(True)
    plt.show()

def plot_singular_values(X, indices=[3, 7, 11]):
 
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    S = S[:102]
    
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Valores singulares $\sigma_i$')

    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot(idx-1, S[idx-1], 'o', color=color, markersize = 4) 
        plt.vlines(x=idx-1, ymin=S[-1], ymax=S[idx-1], color=color, linestyle=style, label=f'd={(idx - 1)}')
        plt.hlines(y=S[idx-1], xmin=0, xmax=idx-1, color=color, linestyle=style)
    
    print (f"Valores singulares para d = 2: {S[1]:.4f}")
    print (f"Valores singulares para d = 6: {S[5]:.4f}")
    print (f"Valores singulares para d = 10: {S[9]:.4f}")
    print (f"Valores singulares para d = 102: {S[101]:.4f}")
    print (f"Valores singulares para d = 1: {S[0]:.4f}")
    
    print (f"Diferencia d = 10 - d = 2: {S[9] - S[1]:.4f}")
    
    
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
    
    # covariance_matrix = compute_covariance_matrix(normalized_dataset)
    
    # plot_covariance_matrix(covariance_matrix)
    
    # plot_matrices(normalized_dataset)
    media_acumulada_valores_singulares(normalized_dataset)
    valores_singulares_acumulada(normalized_dataset)
    
    
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
    valores_singulares_acumulada(normalized_dataset)

    K_X, error = original(normalized_dataset, Y)
    similarity_matrices_og = [K_X]
    titles_og = ["Matriz de Similaridad en el Espacio Original"]
    errors_og = [error]
    
    similarity_matrices, titles, errors = perform_pca(normalized_dataset, Y, errors_og, titles_og, similarity_matrices_og, dims, 10)
    plot_similarity_matrices(similarity_matrices, titles)
    plot_prediction_errors(errors, dims_e, 'PCA')

if __name__ == "__main__":
    main()
