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
delta2_values = [0.01, 0.1, 1, 10]

# Gráficos para diferentes valores de delta2
for delta2 in delta2_values:
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    cost_F2 = [F2(x, delta2) for x in history_F2]
    
    plt.figure()
    plt.plot(cost_F2, label=f'$F_2(x_k)$ con $\\delta_2={delta2}$')
    plt.axhline(y=F2(x_svd, delta2), color='r', linestyle='--', label='$F_2(x^*)$ (SVD)')
    plt.yscale('log')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.legend()
    plt.title(f'Evolución del costo $F_2$ para $\\delta_2={delta2}$')
    plt.savefig(f'costo_evolucion_F2_delta2_{delta2}.png')
    plt.show()

    plt.figure()
    plt.plot(history_F2[-1], label=f'Gradiente Descendente con $\\delta_2={delta2}$')
    plt.plot(x_svd, label='Solución mediante SVD', linestyle='dashed')
    plt.xlabel('Índice')
    plt.ylabel('Valor de x')
    plt.legend()
    plt.title(f'Solución obtenida con $\\delta_2={delta2}$')
    plt.savefig(f'solucion_comparacion_delta2_{delta2}.png')
    plt.show()

# Desviación estándar y norma de la solución obtenida con regularización L2
std_dev = []
norms = []

for delta2 in delta2_values:
    history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2)
    std_dev.append(np.std(history_F2[-1]))
    norms.append(np.linalg.norm(history_F2[-1]))

plt.figure()
plt.plot(delta2_values, std_dev, marker='o')
plt.xlabel('$\\delta_2$')
plt.ylabel('Desviación Estándar')
plt.title('Desviación Estándar de la Solución vs $\\delta_2$')
plt.savefig('desviacion_estandar.png')
plt.show()

plt.figure()
plt.plot(delta2_values, norms, marker='o')
plt.xlabel('$\\delta_2$')
plt.ylabel('Norma de la Solución')
plt.title('Norma de la Solución vs $\\delta_2$')
plt.savefig('norma_solucion.png')
plt.show()

# Evolución de la función de costo F y F2
history_F = gradient_descent(grad_F, x0, step_size, max_iter)
cost_F = [F(x) for x in history_F]
history_F2 = gradient_descent(grad_F2, x0, step_size, max_iter, delta2_values[-1])
cost_F2 = [F2(x, delta2_values[-1]) for x in history_F2]

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(cost_F2, label='$F_2(x_k)$')
plt.axhline(y=F2(x_svd, delta2_values[-1]), color='r', linestyle='--', label='$F_2(x^*)$ (SVD)')
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Evolución del costo $F_2$')

plt.subplot(1, 2, 2)
plt.plot(cost_F, label='$F(x_k)$')
plt.axhline(y=F(x_svd), color='r', linestyle='--', label='$F(x^*)$ (SVD)')
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Evolución del costo $F$')

plt.tight_layout()
plt.savefig('costo_evolucion_F_F2.png')
plt.show()
