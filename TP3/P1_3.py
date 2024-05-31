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

# La solución se obtiene entonces como β = X^+ y = V S^+ U^T y. \\


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from P1_1 import load_data, normalize_dataset, pca_with_svd


# def perform_pca(X_dataset, Y, dims):
#     pca = PCA(n_components=X_dataset.shape[1])
#     pca.fit(X_dataset)
#     V = pca.components_.T
    
#     for d in dims:
#         Vd = V[:, :d]
#         Z = np.dot(X_dataset, Vd)
#         model = LinearRegression().fit(Z, Y)
#         y_pred = model.predict(Z)

#     return model, y_pred

def svd_least_squares(X, y, d):
    # Descomposición en valores singulares (SVD)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
        
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = Vt[:d, :]
    V_d = V[:, :d]
    
    # Calcular la pseudo-inversa de S
    S_inv = np.diag(1 / S_d)

    # Calcular la pseudo-inversa de X
    X_pseudo_inv = V_d @ S_inv @ U_d.T
    
    # PCA
    X_pca = S_inv @ U_d.T

    # Calcular el vector de parámetros beta
    # beta = X_pseudo_inv @ y
    
    beta = X_pca @ y
    
    error = np.linalg.norm(X @ VT_d.T @ beta - y) / np.linalg.norm(y)

    return beta, error



def pca_analysis(X, y, d_values):
    results = {}
    for d in d_values:
        # Aplicar PCA para reducir la dimensionalidad a d componentes
        pca = PCA(n_components=d)
        X_reduced = pca.fit_transform(X)

        # Calcular los parámetros beta usando SVD en el espacio reducido
        beta = svd_least_squares(X_reduced, y)

        # Calcular las predicciones
        y_pred = X_reduced @ beta

        # Calcular el error de predicción
        error = np.linalg.norm(y - y_pred)

        results[d] = {
            "beta": beta,
            "error": error,
            "y_pred": y_pred,
            "explained_variance_ratio": pca.explained_variance_ratio_
        }

    return results


def plot_prediction_errors(X, y):
    errors = []
    dims = range(1, X.shape[1]+1)

    for d in dims:
        beta, error = svd_least_squares(X, y, d)
        errors.append(error)
        print(f"Dimensión: {d}, Error de predicción: {error}")

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
    # X, y = load_data()
    # X_normalized = normalize_dataset(X)

    # # Definir los valores de d a considerar
    # dims = [2, 6, 10, X.shape[1]]
    
    X, y = load_data()
    X = normalize_dataset(X)
    leastSquares(X, y)
    plot_prediction_errors(X, y)

    # # Realizar el análisis de PCA y calcular los parámetros beta
    # results = pca_analysis(X_normalized, y, dims)

    # # Determinar la mejor dimensión d que minimiza el error de predicción
    # best_d = min(results, key=lambda d: results[d]['error'])
    # best_model = results[best_d]

    # print(f"Mejor dimensión d: {best_d}")
    # print(f"Error de predicción: {best_model['error']}")
    # print(f"Vector beta: {best_model['beta']}")
    # print(f"Varianza explicada por componente: {best_model['explained_variance_ratio']}")

    # # Resolver el problema de cuadrados mínimos en el espacio original X
    # beta_original = svd_least_squares(X_normalized, y)
    # print(f"Vector beta en el espacio original: {beta_original}")

    # # Graficar los errores de predicción para cada valor de d
    # errors = [results[d]['error'] for d in dims]
    # plt.figure(figsize=(10, 6))
    # plt.plot(dims, errors, marker='o')
    # plt.title('Error de Predicción vs Dimensión d')
    # plt.xlabel('Dimensión d')
    # plt.ylabel('Error de Predicción')
    # plt.grid(True)
    # plt.show()

    # # Graficar la varianza explicada acumulada para el mejor modelo
    # explained_variance_ratio = np.cumsum(best_model['explained_variance_ratio'])
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, best_d + 1), explained_variance_ratio, marker='o')
    # plt.title('Varianza Explicada Acumulada')
    # plt.xlabel('Número de Componentes Principales')
    # plt.ylabel('Varianza Explicada Acumulada')
    # plt.grid(True)
    # plt.show()

    # # Graficar los pesos del vector beta en el espacio original
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(1, len(beta_original) + 1), beta_original)
    # plt.title('Pesos del Vector β en el Espacio Original')
    # plt.xlabel('Dimensiones Originales')
    # plt.ylabel('Pesos de β')
    # plt.grid(True)
    # plt.show()

    # # Graficar las predicciones vs las observaciones reales para el mejor modelo
    # y_pred_best = best_model['y_pred']
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y, y_pred_best, c = y, cmap='viridis')
    # plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    # plt.title('Predicciones vs Observaciones Reales')
    # plt.xlabel('Observaciones Reales')
    # plt.ylabel('Predicciones')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()