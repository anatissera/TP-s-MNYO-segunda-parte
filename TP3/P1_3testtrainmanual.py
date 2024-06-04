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
    return X_pca, Vt_d

def svd_least_squares_PCA(X, y, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    X_pseudo_inv = Vt.T @ pseudo_inverse(np.diag(S)) @ U.T
    beta = X_pseudo_inv @ y
    error = np.linalg.norm(X @ beta - y) ** 2
    return X_pseudo_inv, beta, error, S

def shuffle_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_set_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def plot_prediction_errors_train_test(X_train, y_train, X_test, y_test):
    test_errors = []
    dims = range(1, X_train.shape[1] + 1)
    
    for d in dims:
        X_train_pca, Vt_d = generate_pca(X_train, d)
        _, beta, train_error, _ = svd_least_squares_PCA(X_train_pca, y_train, d)
        
        X_test_pca = X_test @ Vt_d.T
        test_error = np.linalg.norm(X_test_pca @ beta - y_test) ** 2
        test_errors.append(test_error)
        
    plt.figure(figsize=(12, 6))
    plt.plot(dims, test_errors, 'o-', markersize=2.5, color="darkcyan", linewidth=2, label='Error evaluado en Prueba $\\neq$ Entrenamiento')
    plt.xlabel('Dimensiones', fontsize=14)
    plt.ylabel('Error de predicción cuadrático (norma 2)', fontsize=14)
    plt.title('Error de predicción cuadrático con norma 2 para diferentes dimensiones con entrenamiento y prueba', fontsize=15)
    plt.legend(fontsize= 12)
    plt.grid(False)
    plt.show()
    
    best_dimension = dims[np.argmin(test_errors)]
    print(f"La mejor dimensión es {best_dimension} con un error de prueba de {test_errors[best_dimension-1]}")
    return best_dimension

def plot_beta_weights(beta):
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(beta) + 1), beta)
    plt.title('Pesos del Vector β en el Espacio Original')
    plt.xlabel('Dimensiones Originales')
    plt.ylabel('Pesos de β')
    plt.grid(False)
    plt.show()

def plot_3d(X, y, beta):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', marker='o')
    
    x_surf, y_surf = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), 
                                 np.linspace(X[:, 1].min(), X[:, 1].max(), 100))

    if len(beta) == 3:
        z_surf = beta[0] * x_surf + beta[1] * y_surf + beta[2]
    elif len(beta) == 2: 
        z_surf = beta[0] * x_surf + beta[1] * y_surf
    else:
        raise ValueError()
    
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.6, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.title('Predicción 3D vs Real')
    plt.show()

def plot_predictions_vs_observations_2D(y, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y, y_pred, c=y, cmap='viridis', label='Predicciones')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Observaciones')
    plt.title('Predicciones vs Observaciones Reales')
    plt.xlabel('Observaciones Reales')
    plt.ylabel('Predicciones')
    plt.legend()
    plt.grid(False)
    plt.show()
    
def plot_singular_values(S):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(S) + 1), S, 'o-', markersize=4, color="blue", linewidth=2)
    plt.xlabel('Componentes')
    plt.ylabel('Valor Singular')
    plt.title('Valores Singulares')
    plt.grid(True)
    plt.show()
    
def graficar_y_pred_vs_y_real(y1, y_pred1, y2, y_pred2):
    plt.figure()
    plt.plot(range(len(y1)), y1, 'o', label='Train Real', )
    plt.plot(range(len(y_pred1)), y_pred1, 'o', label='Train Aproximación')
    plt.plot(range(len(y2)), y2, 'o', label='Test Real', )
    plt.plot(range(len(y_pred2)), y_pred2, 'o', label='Test Aproximación')
    plt.title("Comparación entre valores reales y aproximados")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()
    plt.show()
    
    
def main():
    x, y = load_data()
    X = normalize_dataset(x)
    labels = normalize_dataset(y)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = shuffle_split(X, labels, test_size=0.2, random_state=42)
    
    best_dimension = plot_prediction_errors_train_test(X_train, y_train, X_test, y_test)
    
    
    X_train_pca, Vt_d = generate_pca(X_train, 2)
    _, beta, train_error, _ = svd_least_squares_PCA(X_train_pca, y_train, 2)
        
    X_test_pca = X_test @ Vt_d.T
    
    graficar_y_pred_vs_y_real(y_train, X_train_pca @ beta, y_test, X_test_pca @ beta)
    

if __name__ == "__main__":
    main()
