import numpy as np
import matplotlib.pyplot as plt

# Generar matrices y vectores aleatorios
np.random.seed(42)
n = 5
d = 100
A = np.random.rand(n, d)
b = np.random.rand(n)

# Función de costo F(x)
def F(x):
    return np.linalg.norm(A @ x - b)**2

# Función de costo F2(x) con regularización L2
def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

# Gradiente de F(x)
def grad_F(x):
    return 2 * A.T @ (A @ x - b)

# Gradiente de F2(x)
def grad_F2(x, delta2):
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
    return np.array(history)

# Parámetros del algoritmo
x0 = np.random.rand(d)
max_iter = 1000
lambda_max = np.max(np.linalg.eigvals(2 * A.T @ A).real)
step_size = 1 / lambda_max

# Solución mediante SVD
U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
x_svd = VT.T @ np.linalg.inv(np.diag(Sigma)) @ U.T @ b

# Parámetros de regularización
delta2_values = [0.01, 0.1, 1, 10, 100]

# Gráficos para diferentes valores de delta2
colors = ['b', 'g', 'r', 'c', 'm']
markers = ['o', 's', 'd', 'x', '+']

# Evolución del costo F2 para diferentes valores de delta2
plt.figure()
for i, delta2 in enumerate(delta2_values):
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    cost_F2 = [F2(x, delta2) for x in history_F2]
    
    plt.plot(cost_F2, label=f'$F_2(x_k)$ con $\\delta_2={delta2}$', color=colors[i])
    plt.axhline(y=F2(x_svd, delta2), color=colors[i], linestyle='--', label=f'$F_2(x^*)$ con $\\delta_2={delta2}$')
    
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Evolución del costo $F_2$ para diferentes valores de $\\delta_2$')
plt.savefig('costo_evolucion_F2_delta2.png')
plt.show()

# Comparación de la solución obtenida con regularización L2 para diferentes valores de delta2
plt.figure()
for i, delta2 in enumerate(delta2_values):
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    plt.scatter(np.arange(d), history_F2[-1], label=f'$\\delta_2={delta2}$', color=colors[i], marker=markers[i])
plt.scatter(np.arange(d), x_svd, label='Solución mediante SVD', color='k', marker='x')
plt.xlabel('Índice')
plt.ylabel('Valor de x')
plt.legend()
plt.title('Comparación de la solución obtenida con diferentes valores de $\\delta_2$')
plt.savefig('solucion_comparacion_delta2.png')
plt.show()

# Desviación estándar de la solución obtenida con regularización L2 para diferentes valores de delta2
std_dev = []
for delta2 in delta2_values:
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    std_dev.append(np.std(history_F2[-1]))

plt.figure()
plt.plot(delta2_values, std_dev, 'o-')
plt.xlabel('$\\delta_2$')
plt.ylabel('Desviación Estándar')
plt.title('Desviación Estándar de la Solución vs $\\delta_2$')
plt.savefig('desviacion_estandar.png')
plt.show()

# Norma de la solución obtenida con regularización L2 para diferentes valores de delta2
norms = []
for delta2 in delta2_values:
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    norms.append(np.linalg.norm(history_F2[-1]))

plt.figure()
plt.plot(delta2_values, norms, marker='o', linestyle='-')
plt.xlabel('$\\delta_2$')
plt.ylabel('Norma de la Solución')
plt.title('Norma de la Solución vs $\\delta_2$')
plt.savefig('norma_solucion.png')
plt.show()

# Evolución de los valores singulares durante las iteraciones
sigma_evolution = np.zeros((max_iter + 1, len(Sigma)))
for i, delta2 in enumerate(delta2_values):
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
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

# Comparación de los errores relativos entre SVD y las soluciones obtenidas por los diferentes métodos
errors = []
for delta2 in delta2_values:
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    errors.append(np.linalg.norm(x_svd - history_F2[-1]) / np.linalg.norm(x_svd))

history_F = gradient_descent(grad_F, x0, step_size, max_iter)
error_F = np.linalg.norm(x_svd - history_F[-1]) / np.linalg.norm(x_svd)

plt.figure()
plt.bar(['F sin regularizar'] + [f'$\\delta_2={delta2}$' for delta2 in delta2_values], [error_F] + errors, color=colors)
plt.xlabel('Métodos')
plt.ylabel('Error Relativo')
plt.title('Error relativo entre SVD y las soluciones obtenidas')
plt.savefig('error_relativo.png')
plt.show()
