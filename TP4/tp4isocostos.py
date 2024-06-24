import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configuración inicial y parámetros
def configuracion_inicial():
    np.random.seed(0)
    n, d = 5, 100
    A = np.random.rand(n, d)
    b = np.random.rand(n)
    x0 = np.random.rand(d)
    iteraciones = 1000
    sigma_max = np.max(np.linalg.svd(A, full_matrices=False, compute_uv=False))
    lambda_max = 2 * sigma_max**2 # multiplico por 2 porque el hessiano es 2*(A.T @ A) y sigma es la raiz cuadrada de los autovalores de A.T @ A
    paso = 1 / lambda_max
    return A, b, x0, iteraciones, sigma_max, lambda_max, paso

A, b, x0, iteraciones, sigma_max, lambda_max, paso = configuracion_inicial()
n, d = 5, 100

# Funciones de costo y gradiente
def costo(x):
    return np.linalg.norm(A @ x - b)**2

def costo_regularizado(x, delta2):
    return costo(x) + delta2 * np.linalg.norm(x)**2

def gradiente_costo(x):
    return 2 * A.T @ (A @ x - b)

def gradiente_costo_regularizado(x, delta2):
    return gradiente_costo(x) + 2 * delta2 * x

# Algoritmo de gradiente descendente
def descenso_gradiente(x0, iteraciones, paso, regularizacion=False, delta2=0):
    A, b, _, _, _, _, _ = configuracion_inicial()
    x = x0.copy()
    historial_costos = []
    historial_x = [x.copy()]
    for _ in range(iteraciones):
        gradiente = gradiente_costo_regularizado(x, delta2) if regularizacion else gradiente_costo(x)
        x -= paso * gradiente
        historial_costos.append(costo_regularizado(x, delta2) if regularizacion else costo(x))
        historial_x.append(x.copy())
    return x, historial_costos, historial_x

# Gráficos de contorno y trayectorias
def graficar_contorno_y_trayectorias():
    A, b, x0, iteraciones, sigma_max, lambda_max, paso = configuracion_inicial()
    x_svd = np.linalg.pinv(A) @ b
    
    # Generar trayectorias
    _, _, historial_x_f1 = descenso_gradiente(x0, iteraciones, paso)
    _, _, historial_x_f1_2 = descenso_gradiente(x0, iteraciones, paso, delta2=1 * sigma_max)
    _, _, historial_x_f1_3 = descenso_gradiente(x0, iteraciones, paso, delta2=2 * sigma_max)
    
    
    delta = 0.1
    delta2 = delta * sigma_max
    _, _, historial_x_f2 = descenso_gradiente(x0, iteraciones, paso, regularizacion=True, delta2=delta2)
    _, _, historial_x_f2_2 = descenso_gradiente(x0, iteraciones, paso, regularizacion=True, delta2=1 * sigma_max)
    _, _, historial_x_f2_3 = descenso_gradiente(x0, iteraciones, paso, regularizacion=True, delta2=2 * sigma_max)
    
    # PCA
    pca_f1 = PCA(n_components=2).fit(historial_x_f1)
    historial_transformado_f1 = pca_f1.transform(historial_x_f1) 
    historial_transformado_f2 = pca_f1.transform(historial_x_f2)
    historial_transformado_f2_2 = pca_f1.transform(historial_x_f2_2)
    historial_transformado_f2_3 = pca_f1.transform(historial_x_f2_3)
    solucion_svd = pca_f1.transform([x_svd])[0]

    # Gráfico de contorno
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x, y)
    
    mesh = np.array([xx.ravel(), yy.ravel()]).T
    zz = np.array([costo_regularizado(pca_f1.inverse_transform(p), delta2) for p in mesh]).reshape(xx.shape)
    
    
    # Gráfico 1: Trayectoria con F(x)
    plt.figure(figsize=(10, 8))
    plt.plot(historial_transformado_f1[:, 0], historial_transformado_f1[:, 1], 'r-o', markersize=3, linewidth=2, label='Trayectoria GD con $F(x)$', color='lightcoral')
    plt.contourf(xx, yy, zz, levels=50, cmap='cividis', alpha=0.25)
    plt.contour(xx, yy, zz, levels=50, cmap='cividis')
    plt.scatter(solucion_svd[0], solucion_svd[1], color='steelblue', label='Solución SVD', s=100)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Camino del Gradiente Descendente en el Contorno de Isocostos', fontsize=15)
    plt.legend()
    plt.show()
    
    # Gráfico 2: Trayectorias con F2(x) y diferentes deltas
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, zz, levels=50, cmap='cividis', alpha=0.25)
    plt.contour(xx, yy, zz, levels=50, cmap='cividis')
    plt.plot(historial_transformado_f2_2[:, 0], historial_transformado_f2_2[:, 1], 'b-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = 1 \cdot \sigma_{{max}}$', color='seagreen')
    plt.plot(historial_transformado_f2_3[:, 0], historial_transformado_f2_3[:, 1], 'm-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = 5 \cdot \sigma_{{max}}$', color='cadetblue')
    plt.plot(historial_transformado_f2[:, 0], historial_transformado_f2[:, 1], 'b-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = {delta} \cdot \sigma_{{max}}$', color='palevioletred')
    plt.scatter(solucion_svd[0], solucion_svd[1], color='steelblue', label='Solución SVD', s=100)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Camino del Gradiente Descendente en el Contorno de Isocostos', fontsize=15)
    plt.legend()
    plt.show()

    step_sizes = [0.1, 1 ,1.5]

    plt.figure (figsize=(10, 8))
    plt.contourf(xx, yy, zz, levels=50, cmap='cividis', alpha=0.25)
    plt.contour(xx, yy, zz, levels=50, cmap='cividis')
    for step_size in step_sizes:
        step1 = step_size / lambda_max
        _, _, history = descenso_gradiente(x0, iteraciones, step1)
        history_transformed = pca_f1.transform(history)
        plt.plot(history_transformed[:, 0], history_transformed[:, 1], 'o-', markersize=3, linewidth=2, label=f'Trayectoria GD con step = {step_size} $\cdot \lambda_{{max}}^{{-1}}$')
    plt.scatter(solucion_svd[0], solucion_svd[1], color='steelblue', label='Solución SVD', s=100)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Camino del Gradiente Descendiente en el Contorno de Isocostos', fontsize=15)
    plt.legend()
    plt.show()

graficar_contorno_y_trayectorias()

