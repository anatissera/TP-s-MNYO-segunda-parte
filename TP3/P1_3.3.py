import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    df = pd.read_csv("TP3/dataset02.csv", header=None, skiprows=1)
    df = df.iloc[:, 1:]
    X = df.to_numpy()
    labels = np.loadtxt('tp3/y.txt')
    return X, labels

def normalize_dataset(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)

def pseudo_inverse(S_d):
    S_d_inv = np.copy(S_d)
    for i in range(len(S_d)):
        if S_d[i, i] != 0:
            S_d_inv[i, i] = 1 / S_d[i, i]
        else:
            S_d_inv[i, i] = 0
    return S_d_inv

def generate_pca(X, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    X_pca = U_d @ S_d
    return X_pca, U_d, S_d, Vt_d

def svd_least_squares_PCA(X, y, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    X_pseudo_inv = Vt_d.T @ pseudo_inverse(S_d) @ U_d.T
    beta = X_pseudo_inv @ y
    error = np.linalg.norm(X @ beta - y) ** 2
    return X_pseudo_inv, beta, error

def plot_prediction_errors(X, y):
    errors = []
    dims = range(1, X.shape[1] + 1)
    for d in dims:
        A_d, _, _, _ = generate_pca(X, d)
        _, _, error = svd_least_squares_PCA(A_d, y, d)
        errors.append(error)
    best_dimension = dims[np.argmin(errors)]
    plt.figure(figsize=(12, 6))
    plt.plot(dims, errors, 'o-', markersize=2.5, color="darkcyan", linewidth=2)
    plt.xlabel('Dimensiones')
    plt.ylabel('Error de predicción (norma 2)')
    plt.title('Error de predicción para diferentes dimensiones')
    plt.grid(True)
    plt.show()
    print(f"La mejor dimensión es {best_dimension} con un error de {errors[best_dimension-1]}")
    return best_dimension

def plot_beta_weights(beta):
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(beta) + 1), beta)
    plt.title('Pesos del Vector β en el Espacio Original')
    plt.xlabel('Dimensiones Originales')
    plt.ylabel('Pesos de β')
    plt.grid(True)
    plt.show()

def plot_3d(X, y, beta):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of the original data points
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', marker='o')
    
    # Create a grid to plot the plane
    x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
                                 np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    z_surf = beta[0] * x_surf + beta[1] * y_surf + beta[2]
    
    # Plot the plane
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.title('Predicción 3D vs Real con PCA')
    plt.show()

def main():
    x, y = load_data()
    X = normalize_dataset(x)
    labels = normalize_dataset(y)
    
    best_dimension = plot_prediction_errors(X, labels)
    
    X_pca, U_d, S_d, Vt_d = generate_pca(X, best_dimension)
    X_pseudo_inv, beta, error = svd_least_squares_PCA(X, labels, best_dimension)
    
    plot_beta_weights(beta)
    
    
    beta = np.append(beta, 0)  # Assuming no intercept in the original regression
    plot_3d(X_pca, labels, beta)

if __name__ == "__main__":
    main()
