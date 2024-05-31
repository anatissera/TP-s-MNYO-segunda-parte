import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from P1_1 import load_data, normalize_dataset, pca_with_svd

def pseudo_inverse(S_d):
    S_d_inv = np.copy(S_d)
    
    for i in range(len(S_d - 4)):
        if S_d[i, i] != 0:
            S_d_inv[i, i] = 1 / S_d[i, i]
        else:
            S_d_inv[i, i] = 0
            
    for i in range (len(S_d - 4), len(S_d)):
        S_d_inv[i, i] = 0
    
    return S_d_inv

# valor singular para d = 103 0.0000
# valor singular para d = 104 0.0000
# valor singular para d = 105 0.0000
# valor singular para d = 106 0.0000

def svd_least_squares_PCA(X, y, d):
    # X = normalize_dataset(X)
    # Descomposición en valores singulares (SVD)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    # Seleccionar las primeras d dimensiones
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    V_d = V[:, :d]
    
    # Calcular la pseudo-inversa de S_d
    # S_inv = np.diag(1 / S_d)
    
    # Calcular la pseudo-inversa de X reducida
    # X_d_pseudo_inv = V_d @ S_inv @ U_d.T

    # X_pca = np.linalg.pinv(U_d @ S_d) 
    # X_pca = (np.linalg.inv(S_d)) @ U_d.T
    X_pca = pseudo_inverse(S_d) @ U_d.T
    
    # Calcular el error de predicción
    beta = X_pca @ y
    
    error = np.linalg.norm(X @ Vt_d.T @ beta - y) / np.linalg.norm(y)
    
    return beta, error


# def plot_prediction_errors(X, y):
#     errors = []
#     dims = range(1, X.shape[1] + 1)

#     for d in dims:
#         beta, error = svd_least_squares(X, y, d)
#         errors.append(error)
       
#     best_dimension = dims[np.argmin(errors)]   
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(dims, errors, 'o-', markersize=2.5, color="darkcyan", linewidth=y)
#     plt.xlabel('Dimensiones')
#     plt.ylabel('Error de predicción (norma 2)')
#     plt.title('Error de predicción para diferentes dimensiones')
#     plt.grid(True)
#     plt.show()
    
#     print(f"La mejor dimensión es {best_dimension} con un error de {errors[best_dimension-1]}")


def svd_least_squares(X, y, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    
    X_pca_pseudo_inv = Vt_d.T @ pseudo_inverse(S_d) @ U_d.T
    beta = X_pca_pseudo_inv @ y
    
    error = np.linalg.norm(X @ beta - y) / np.linalg.norm(y)
    
    return beta, error

def plot_prediction_errors(X, y, bool=False):
    errors = []
    dims = range(1, X.shape[1] + 1)
    
    for d in dims:
        if bool:
            _, error = svd_least_squares(X, y, d)
        else:
            _, error = svd_least_squares_PCA(X, y, d)
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

def plot_predictions_vs_observations(y, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y, y_pred, c= y, cmap='viridis', label='Predicciones')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Observaciones')
    plt.title('Predicciones vs Observaciones Reales')
    plt.xlabel('Observaciones Reales')
    plt.ylabel('Predicciones')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def main():
    X, y = load_data()
    X = normalize_dataset(X)
    plot_prediction_errors(X, y)
    
    best_dimension = plot_prediction_errors(X, y, True)
    
    beta, _ = svd_least_squares(X, y, best_dimension)
    y_pred = X @ beta
    
    plot_beta_weights(beta)
    plot_predictions_vs_observations(y, y_pred)


if __name__ == '__main__':
    main()