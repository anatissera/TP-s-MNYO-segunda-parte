# Los datos X vienen acompañados de una variable dependiente respuesta o etiquetas llamada Y (archivo
# y.txt) estructurada como un vector n × 1 para cada muestra. Queremos encontrar el vector β y modelar
# linealmente el problema que minimice la norma
# ∥Xβ − y∥_2
# de manera tal de poder predecir con Xβ = y lo mejor posible a las etiquetas y, es decir, minimizar el
# error de predicción. Usando PCA, que dimensión d mejora la predicción? Cuales muestras son las de
# mejor predicción con el mejor modelo? Resolviendo el problema de cuadrados mínimos en el espacio
# original X, que peso se le asigna a cada dimensión original si observamos el vector β?


# Para resolver el problema de cuadrados mínimos Xβ= y utilizando SVD, primero se calcula la descomposición en valores singulares de X. 
# Una vez obtenida la descomposición, la pseudo-inversa de X, denotada como X^+, se utiliza para encontrar la solución de cuadrados mínimos. \\
# Esta se define como X^+ = V S^+ U^t donde S^+ es la pseudo-inversa de S. \\
# Si S es una matriz diagonal con elementos s_i en su diagonal, entonces S^+ es una matriz diagonal con elementos 
# 1/ s_i si s_i != 0, y 0 en caso contrario.

# La solución se obtiene entonces como β = X^+ y = V S^+ U^T y. \\                               def load_data():

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

def generate_pca(X, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    V_d = V[:, :d]
    
    X_pca = U_d @ S_d
    
    return X_pca

def svd_least_squares_PCA(X, y, d):
   
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    V_d = V[:, :d]
    
    X_pseudo_inv = Vt_d.T @ pseudo_inverse(S_d) @ U_d.T
    
    beta = X_pseudo_inv @ y
    
    # error norma 2 al cuadrado
    error = np.linalg.norm(X @ beta - y) **2
    
    return X_pseudo_inv, beta, error


def plot_prediction_errors(X, y, bool=False):
    errors = []
    dims = range(1, X.shape[1] + 1)
    
    for d in dims:
        A_d = generate_pca(X, d)
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

def main():
    X, y = load_data()
    X = normalize_dataset(X)
    y = normalize_dataset(y)
    

    plot_prediction_errors(X, y)
    
    X_pseudo_inv, beta, error = svd_least_squares_PCA(X, y, 2)
    plot_beta_weights(beta)
    
    X_pseudo_inv, beta, error = svd_least_squares_PCA(X, y, 3)
    plot_beta_weights(beta)
    
    X_pseudo_inv, beta, error = svd_least_squares_PCA(X, y, 103)
    plot_beta_weights(beta)
    
    return
    
if __name__ == "__main__":
    main()