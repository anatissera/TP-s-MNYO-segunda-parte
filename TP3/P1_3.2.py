import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from P1_1 import load_data, normalize_dataset, pca_with_svd

def svd_least_squares(X, y, d):
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
    
    # Calcular el vector de parámetros beta

   
    X_pca = np.linalg.pinv(U_d @ S_d) 
    # X_pca = (np.linalg.inv(S_d)) @ U_d.T
    
    
    # Calcular el error de predicción
    beta = X_pca @ y
    
    error = np.linalg.norm(X @ Vt_d.T @ beta - y) / np.linalg.norm(y)
    
    return beta, error

def plot_prediction_errors(X, y):
    errors = []
    dims = range(1, X.shape[1] + 1)

    for d in dims:
        beta, error = svd_least_squares(X, y, d)
        errors.append(error)
        # print(f"Dimensión: {d}, Error de predicción: {error}")

    plt.figure(figsize=(12, 6))
    plt.plot(dims, errors, marker='o')
    plt.xlabel('Dimensiones')
    plt.ylabel('Error de predicción (norma 2)')
    plt.title('Error de predicción para diferentes dimensiones')
    plt.grid(True)
    plt.show()
    
    
def leastSquares(matrix, y):
    # Realizamos la descomposición SVD
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    errors = []
    dimensions = []

    for d in range(1, matrix.shape[1] + 1):
        # Realizamos la reducción de dimensiones
        U_reduced = u[:, :d]
        S_reduced = np.diag(s[:d])
        V_reduced = v[:d, :]
        matrix_svd = U_reduced @ S_reduced @ V_reduced
        
        # Calculamos la matriz de coeficientes
        A = np.linalg.inv(S_reduced) @ U_reduced.T
        # A = np.linalg.pinv(U_reduced @ S_reduced)
        
        # Calculamos los coeficientes
        beta = A @ y
        
        # Calculamos el error
        error = np.linalg.norm(matrix @ V_reduced.T @ beta - y) / np.linalg.norm(y)

        errors.append(error)
        dimensions.append(d)
    
    best_dimension = dimensions[np.argmin(errors)]    
            
    plt.figure()
    plt.plot(dimensions, errors)
    plt.title("Error relativo entre normas en función de la dimensión")
    plt.xlabel("Dimensión")
    plt.ylabel("% de Error")
    # plt.ylim(0.75, 0.80)
    plt.show()
    
    print(f"La mejor dimensión es {best_dimension} con un error de {errors[best_dimension-1]}")
    

def main():
    X, y = load_data()
    X = normalize_dataset(X)
    leastSquares(X, y)
    # plot_prediction_errors(X, y)

if __name__ == '__main__':
    main()