import numpy as np
import matplotlib.pyplot as plt

# Generar matrices y vectores aleatorios
np.random.seed(0)
n = 5
d = 100
A = np.random.rand(n, d)
b = np.random.rand(n)


# # Configurations
# n = 2  # Problem dimension
# d = 2  # Parameter space dimension
# iterations = 100
# delta2 = 1e-2

# # Random matrices and vectors generation
# np.random.seed(0) 
# A = np.random.randn(n, d)
# b = np.random.randn(n)

iterations = 2000

sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
sigma_max = np.max(sigma)

# HF1(x) = 2 * A.T @ A
lambda_max = 2 * sigma_max**2 # multiplico por 2 porque el hessiano es 2*(A.T @ A) y sigma es la raiz cuadrada de los autovalores de A.T @ A

delta_constants = [0.001, 0.01, 0.1, 1, 10]  # Constante de regularización

delta2 = 1e-2 * sigma_max

# Paso de aprendizaje
step = 1 / lambda_max


def F(x):
    """Calcula la función de costo F(x) = (Ax - b)^T (Ax - b)."""
    return np.linalg.norm(A @ x - b)**2

def grad_F(x):
    """Calcula el gradiente de F(x)."""
    return 2 * A.T @ (A @ x - b)

def hessiano_F(A): # es la segunda derivada de F(x) porque depende de una sola variable
    """Calcula el Hessiano de F(x)."""
    return 2 * A.T @ A 

# Función de costo F2(x) con regularización L2
def F2(x, delta2):
    """Calcula la función de costo F2(x) = F(x) + delta2 * ||x||^2.
    """
    return F(x) + delta2 * np.linalg.norm(x)**2

# Gradiente de F2(x)
def grad_F2(x, delta2):
    """Calcula el gradiente de F2(x)."""
    return grad_F(x) + 2 * delta2 * x

# Gradiente descendente
def gradient_descent(grad_func, x0, step_size, max_iter, delta2=None):
    x = x0
    history = [x0]
    for _ in range(max_iter):
        if delta2 is not None:
            x = x - step_size * grad_func(x, delta2)
        else:
            x = x - step_size * grad_func(x)
        history.append(x)
    return x, np.array(history)



def SVD(A, b):
    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
    x_svd = VT.T @ np.linalg.inv(np.diag(Sigma)) @ U.T @ b
    return x_svd

thickness = 2.5
def plotF(x_svd, x0, step, iterations):
    plt.figure()
    _, history_f1 = gradient_descent(grad_F, x0, step, iterations)
    x2, history_f2 = gradient_descent(grad_F2, x0, step,  iterations, delta2=delta2)
    cost_F2 = [F2(x, delta2) for x in history_f2]
    cost_f = [F(x) for x in history_f1]

    # plt.plot(cost_f, linewidth=1.7, label="$F(x)$", color = "cadetblue")
    plt.plot(cost_F2, linewidth=thickness, label="$F_2(x)$ con $\delta_2 =$ $10^{-2}$ $\cdot \sigma_{max}$", color= "lightcoral")
    # plt.hlines(F(x_svd), 0, iterations, colors='darkslateblue', linestyles='dashed', label='$F(x)$ de la solución con SVD', linewidth=thickness)
    plt.hlines( delta2 * np.linalg.norm(x2)**2, 0, iterations, colors='darkred', linestyles='dashed', label='$\delta_2 \cdot ||x||^2$', linewidth=thickness)
    
    plt.xlabel('Iteraciones', fontsize=15)
    plt.ylabel('Valor de las funciones (esc log)', fontsize=15)
    plt.legend(fontsize=14)  # Increase the font size to make the legend box bigger
    plt.yscale('log')
    plt.title('Evolución de $F_2(x)$ por iteración', fontsize=20)
    plt.grid(False)
    plt.show()




def plot_costo_F2(x_svd, x0, step_size, max_iter, delta2_values, colors):
  
    plt.figure()
    for i, delta in enumerate(delta2_values):
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        cost_F2 = [F2(x, delta2) for x in history_F2]
        
        plt.plot(cost_F2, label=f'$F_2(x_k)$ con $\\delta_2={delta}$$\cdot \sigma_{{max}}$', color=colors[i])
        # plt.axhline(y=F2(x_svd, delta2), color=colors[i], linestyle='--', label=f'$F_2(x^*)$ con $\\delta_2={delta2}$')
        
    plt.yscale('log')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.legend()
    plt.title('Evolución del costo $F_2$ para diferentes valores de $\\delta_2$')
    plt.savefig('costo_evolucion_F2_delta2.png')
    plt.show()

def L2(x_svd, x0, step_size, max_iter, delta2_values, d=100, colors=['b', 'g', 'r', 'c', 'm'], markers=['o', 's', 'd', 'x', '+']):
    # Comparación de la solución obtenida con regularización L2 para diferentes valores de delta2
    plt.figure()
    for i, delta in enumerate(delta2_values):
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        cost_F2 = [F2(x, delta2) for x in history_F2]
        
        plt.scatter(np.arange(d), history_F2[-1], label=f'$\\delta_2={delta}$$\cdot \sigma_{{max}}$', color=colors[i], marker=markers[i], alpha = 0.9)
    plt.scatter(np.arange(d), x_svd, label='Solución mediante SVD', color='k', marker='x')
    plt.xlabel('Índice')
    plt.ylabel('Valor de x')
    plt.legend()
    plt.title('Comparación de la solución obtenida con diferentes valores de $\\delta_2$')
    plt.savefig('solucion_comparacion_delta2.png')
    plt.show()

def sigma_L2(x0, step_size, max_iter, delta2_values):
    # Desviación estándar de la solución obtenida con regularización L2 para diferentes valores de delta2
    std_dev = []
    for delta in delta2_values:
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        std_dev.append(np.std(history_F2[-1]))

    plt.figure()
    plt.plot(delta2_values, std_dev, 'o-')
    plt.xlabel('$\\delta_2$')
    plt.ylabel('Desviación Estándar')
    plt.title('Desviación Estándar de la Solución vs $\\delta_2$')
    plt.savefig('desviacion_estandar.png')
    plt.show()

def norma_L2(x0, step_size, max_iter, delta2_values):
    # Norma de la solución obtenida con regularización L2 para diferentes valores de delta2
    norms = []
    for delta in delta2_values:
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        norms.append(np.linalg.norm(history_F2[-1]))

    plt.figure()
    plt.plot(delta2_values, norms, marker='o', linestyle='-')
    plt.xlabel('$\\delta_2$')
    plt.ylabel('Norma de la Solución')
    plt.title('Norma de la Solución vs $\\delta_2$')
    plt.savefig('norma_solucion.png')
    plt.show()

def singular_values(A, x0, step_size, max_iter, delta2_values, Sigma):
    # Evolución de los valores singulares durante las iteraciones
    sigma_evolution = np.zeros((max_iter + 1, len(Sigma)))
    for i, delta in enumerate(delta2_values):
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        for j in range(max_iter + 1):
            sigma_evolution[j, :] = np.linalg.svd(A @ history_F2[j].reshape(-1, 1), compute_uv=False)

    plt.figure()
    for i in range(len(Sigma)):
        plt.plot(sigma_evolution[:, i], label=f'$\\sigma_{i+1}$')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valores singulares')
    plt.title('Evolución de los valores singulares')
    plt.legend()
    plt.savefig('sigma_evolucion.png')
    plt.show()

def error_relativo(x_svd, x0, step_size, max_iter, delta2_values, colors= ['lightcoral', 'peachpuff', 'darkseagreen', 'lightseagreen', 'powderblue', 'mediumslateblue']):
    # Comparación de los errores relativos entre SVD y las soluciones obtenidas por los diferentes métodos
    errors = []
    for delta in delta2_values:
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        errors.append(np.linalg.norm(x_svd - history_F2[-1]) / np.linalg.norm(x_svd))

    history_F = gradient_descent(grad_F, x0, step_size, max_iter)
    error_F = np.linalg.norm(x_svd - history_F[-1]) / np.linalg.norm(x_svd)

    plt.figure()
    plt.bar(['F sin regularizar'] + [f'$\\delta_2={delta}$$\cdot \sigma_{{max}}$' for delta in delta2_values], [error_F] + errors, color=colors)
    plt.xlabel('Métodos')
    plt.ylabel('Error Relativo')
    plt.title('Error relativo entre SVD y las soluciones obtenidas')
    plt.savefig('error_relativo.png')
    plt.show()
    
def calc_errors(x_svd, x0, step_size, max_iter, delta_constants):
    errors_relativos = []
    errores_absolutos = []
    for delta in delta_constants:
        delta2 = delta * sigma_max
        _, history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
        errors_relativos.append(np.linalg.norm(x_svd - history_F2[-1]) / np.linalg.norm(x_svd))
        errores_absolutos.append(np.abs(F2(history_F2[-1], delta2) - F2(x_svd, delta2)))
    
    history_F = gradient_descent(grad_F, x0, step_size, max_iter)
    error_relativo_F = np.linalg.norm(x_svd - history_F[-1]) / np.linalg.norm(x_svd)
    error_absoluto_F = np.abs(F(history_F[-1]) - F(x_svd))
    
    return error_relativo_F, errors_relativos, error_absoluto_F, errores_absolutos

# def calc_errors(x_svd, x0, step_size, max_iter, delta_constants):
#     errors_relativos = []
#     errores_absolutos = []
#     for delta in delta_constants:
#         delta2 = delta * sigma_max
#         history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
#         errors_relativos.append(np.linalg.norm(F2(x_svd, delta2) - F2(history_F2[-1], delta2)) / np.linalg.norm(F2(x_svd, delta2)))
#         errores_absolutos.append(np.abs(F2(history_F2[-1], delta2) - F2(x_svd, delta2)))
    
#     history_F = gradient_descent(grad_F, x0, step_size, max_iter)
#     error_relativo_F = np.linalg.norm(F(x_svd) - F(history_F[-1])) / np.linalg.norm(F(x_svd))
#     error_absoluto_F = np.abs(F(history_F[-1]) - F(x_svd))
    
#     return error_relativo_F, errors_relativos, error_absoluto_F, errores_absolutos


def plot_errors_r(error_F, errors, delta2_values, colors= ['lightcoral', 'peachpuff', 'darkseagreen', 'lightseagreen', 'powderblue', 'mediumslateblue']):
    plt.figure()
    plt.bar(['F sin regularizar'] + [f'$\\delta_2={delta}$$\cdot \sigma_{{max}}$' for delta in delta2_values], [error_F] + errors, color=colors)
    plt.xlabel('Métodos', fontsize= 15)
    plt.ylabel('Error Relativo', fontsize=15)
    plt.title(r'Error relativo entre SVD y las soluciones obtenidas $\frac{||X - X_{SVD} ||_2}{||X_{SVD}||_2}$', fontsize=16)
    plt.savefig('error_relativo.png')
    plt.yscale('log')
    plt.show()
    
def plot_errors_a(error_F, errors, delta2_values, colors= ['lightcoral', 'peachpuff', 'darkseagreen', 'lightseagreen', 'powderblue', 'mediumslateblue']):
    plt.figure()
    tick_labels = ['$F(x)$', '$F_2(x), \delta^2 = 0.001$', '$F_2(x), \delta^2 = 0.01$', '$F_2(x), \delta^2 = 0.1$', '$F_2(x), \delta^2 = 1$', '$F_2(x), \delta^2 = 10$']
    plt.bar(['F sin regularizar'] + [f'$\\delta_2={delta}$$\cdot \sigma_{{max}}$' for delta in delta2_values], [error_F] + errors, color=colors)
    plt.xlabel('Métodos')
    plt.title('Error absoluto entre SVD y las soluciones obtenidas')
    plt.ylabel('Error absoluto de $||A \cdot x - b||_2$ esc= log', fontsize=15)
    plt.legend(fontsize=14)
    plt.yscale('log')
    plt.show()
    
    
def main():


    # Parámetros del algoritmo
    x0 = np.random.rand(d)
    
    # x0 = np.random.randn(d)
    # x0 = np.random.uniform(0, 1, d)



    max_iter = 1000
    iterations = 1000

    sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    sigma_max = np.max(sigma)

    # HF1(x) = 2 * A.T @ A
    lambda_max = 2 * sigma_max**2 # multiplico por 2 porque el hessiano es 2*(A.T @ A) y sigma es la raiz cuadrada de los autovalores de A.T @ A

    delta_constants = [0.001, 0.01, 0.1, 1, 10]  # Constante de regularización

    delta2 = 1e-2 * sigma_max

    # Paso de aprendizaje
    step_size = 1 / lambda_max
    
    x_svd = SVD(A, b)
    
    plotF(x_svd, x0, step, iterations)
    # Gráficos para diferentes valores de delta2
    colors = ['lightcoral', 'peachpuff', 'seagreen', 'cadetblue', 'midnightblue']
    markers = ['o', 's', 'd', 'x', '+']

    # plot_costo_F2(x_svd, x0, step_size, max_iter, delta_constants, colors)
    # L2(x_svd, x0, step_size, max_iter, delta_constants, d, colors, markers)
    # sigma_L2(x0, step_size, max_iter, delta_constants)
    # norma_L2(x0, step_size, max_iter, delta_constants)
    # singular_values(A, x0, step_size, max_iter, delta_constants, np.linalg.svd(A, full_matrices=False)[1])
    # error_relativo(x_svd, x0, step_size, max_iter, delta_constants)
    
    error_relativo_F, errors_relativos, error_absoluto_F, errores_absolutos = calc_errors(x_svd, x0, step_size, iterations, delta_constants)
    plot_errors_r(error_relativo_F, errors_relativos, delta_constants)
    plot_errors_a(error_absoluto_F, errores_absolutos, delta_constants)
    
    # Graficar errores relativos
    plt.figure()
    plt.bar(['F sin regularizar'] + [f'$\\delta_2={delta}$' for delta in delta_constants], [error_relativo_F] + errors_relativos)
    plt.xlabel('Métodos')
    plt.ylabel('Error Relativo')
    plt.yscale('log')
    plt.title('Error relativo entre SVD y las soluciones obtenidas')
    plt.show()

    # Graficar errores absolutos
    plt.figure()
    plt.bar(['F sin regularizar'] + [f'$\\delta_2={delta}$' for delta in delta_constants], [error_absoluto_F] + errores_absolutos)
    plt.xlabel('Métodos')
    plt.ylabel('Error Absoluto')
    plt.yscale('log')
    plt.title('Error absoluto entre SVD y las soluciones obtenidas')
    plt.show()
if __name__ == '__main__':
    main()