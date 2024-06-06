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

def plot_prediction_errors(X, y):
    errors = []
    dims = range(1, X.shape[1] + 1)
    for d in dims:
        A_d, _ = generate_pca(X, d)
        _, _, error, _ = svd_least_squares_PCA(A_d, y, d)
        errors.append(error)
    best_dimension = dims[np.argmin(errors)]
    plt.figure(figsize=(12, 6))
    plt.plot(dims, errors, 'o-', markersize=2.5, color="darkcyan", linewidth=2)
    plt.xlabel('Dimensiones', fontsize=16)
    plt.ylabel('Error de predicción cuadrático (norma 2)', fontsize=16)
    plt.title('Error de predicción cuadrático para diferentes dimensiones', fontsize=20)
    plt.grid(False)
    plt.show()
    print(f"La mejor dimensión es {best_dimension} con un error de {errors[best_dimension-1]}")
    print (f"El error de predicción para d = 2 es {errors[1]}")
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
    plt.scatter(y, y_pred, c= y, cmap='viridis', label='Predicciones')
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
    
def graficar_y_pred_vs_y_real(y, y_pred):
    plt.figure()
    plt.plot(range(len(y)), y, 'o', label='Real', )
    plt.plot(range(len(y_pred)), y_pred, 'o', label='Aproximación')
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
    
    best_dimension = plot_prediction_errors(X, labels)
   
    X_pca_3, _ = generate_pca(X, 3)
    X_pseudo_inv, beta_3, error, S_3 = svd_least_squares_PCA(X_pca_3, labels, 3)
    
    plot_singular_values(S_3)
    
    plot_3d(X_pca_3, labels, beta_3)
    
    X_pca_2, Vt_d_2 = generate_pca(X, 2)
    X_pseudo_inv_2, beta_2, error_2, S_2 = svd_least_squares_PCA(X_pca_2, labels, 2)
    
    y_pred = X_pca_2 @ beta_2
    
    # ahora quiero ver fila a fila los valores reales y los predichos, cuál minimiza la resta
    #plotear y_pred - labels fila a fila
    diferencia = y_pred - labels
    minimo = np.argmin(np.abs(diferencia))
    print(f"El mínimo error de predicción es {diferencia[minimo]} en la muestra {minimo}")
    
    _, beta, train_error, _ = svd_least_squares_PCA(X, labels, X.shape[1])
    
    # printear en orden descendiente el peso y su dimensión correspondiente
    sorted_weights = sorted(enumerate(beta), key=lambda x: -abs(x[1]))
    peso_ruido = 0
    for i, weight in sorted_weights[6:]:
        peso_ruido += np.abs(weight)
        print(f"Dimensión {i+1}: {weight}")
    print(f"El peso del ruido es {peso_ruido}")
    heaviest_dimension = np.argmax(np.abs(beta))
    print(heaviest_dimension)
    plot_beta_weights(beta)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(labels)), labels, label='Valores reales', color='blue')
    plt.scatter(range(len(y_pred)), y_pred, label='Valores predichos', color='red')
    plt.xlabel('Fila')
    plt.ylabel('Valor')
    plt.title('Valores reales vs Valores predichos')
    plt.legend()
    plt.show()

    # Graficamos la diferencia entre los valores predichos y los reales
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(diferencia)), diferencia)
    plt.xlabel('Fila')
    plt.ylabel('Diferencia')
    plt.title('Diferencia entre valores predichos y reales')
    plt.show() 
    
    graficar_y_pred_vs_y_real(labels, X_pca_2 @ beta_2)
    
    plot_beta_weights(beta_2)
    
    # proyecto X a 2d
    X_proj = X @ Vt_d_2.T
    y_pred_2D = X_proj @ beta_2
    
    plot_predictions_vs_observations_2D(labels, y_pred_2D)    

if __name__ == "__main__":
    main()