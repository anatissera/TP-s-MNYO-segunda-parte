import numpy as np
import matplotlib.pyplot as plt

# Función para calcular el gradiente
def calcular_gradiente(A, x, b):
    return 2 * A.T @ (A @ x - b)

# Función para calcular la función de costo F
def costo_F(A, x, b):
    return np.linalg.norm(A @ x - b)**2

# Función para calcular la función de costo F2 con regularización L2
def costo_F2(A, x, b, delta2):
    return costo_F(A, x, b) + delta2 * np.linalg.norm(x)**2

# Gradiente descendente
def gradiente_descendente(A, b, x_inicial, s, iteraciones):
    x = x_inicial
    costos = []
    for _ in range(iteraciones):
        gradiente = calcular_gradiente(A, x, b)
        x = x - s * gradiente
        costos.append(costo_F(A, x, b))
    return x, costos

# Gradiente descendente con regularización L2
def gradiente_descendente_L2(A, b, x_inicial, s, iteraciones, delta2):
    x = x_inicial
    costos = []
    for _ in range(iteraciones):
        gradiente = calcular_gradiente(A, x, b) + 2 * delta2 * x
        x = x - s * gradiente
        costos.append(costo_F2(A, x, b, delta2))
    return x, costos

# Parámetros del problema
n = 5
d = 100
np.random.seed(0)
A = np.random.randn(n, d)
b = np.random.randn(n)
x_inicial = np.random.randn(d)
sigma_max = np.linalg.norm(A, 2)
delta2 = 10**(-2) * sigma_max
iteraciones = 1000

# Calcular autovalores de A^T A para determinar s
H = A.T @ A
autovalores = np.linalg.eigvals(H)
lambda_max = np.max(autovalores)
s = 1 / lambda_max

# Resolver el problema con gradiente descendente
x_gd, costos_gd = gradiente_descendente(A, b, x_inicial, s, iteraciones)

# Resolver el problema con gradiente descendente y regularización L2
x_gd_L2, costos_gd_L2 = gradiente_descendente_L2(A, b, x_inicial, s, iteraciones, delta2)

# Solución mediante SVD
U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
x_svd = VT.T @ np.linalg.pinv(np.diag(Sigma)) @ U.T @ b

# Graficar la evolución del costo
plt.figure(figsize=(10, 6))
plt.plot(costos_gd, label='Gradiente Descendente')
plt.plot(costos_gd_L2, label='Gradiente Descendente con L2')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Evolución del Costo')
plt.savefig('costo_evolucion.png')
plt.show()

# Graficar las soluciones obtenidas
plt.figure(figsize=(10, 6))
plt.plot(x_gd, label='Gradiente Descendente')
plt.plot(x_gd_L2, label='Gradiente Descendente con L2')
plt.plot(x_svd, label='Solución mediante SVD', linestyle='dashed')
plt.xlabel('Índice')
plt.ylabel('Valor de x')
plt.legend()
plt.title('Comparación de Soluciones')
plt.savefig('soluciones_comparacion.png')
plt.show()
