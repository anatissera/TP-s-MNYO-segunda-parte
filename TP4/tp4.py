import numpy as np
import matplotlib.pyplot as plt

# Definición de funciones de costo y gradientes
def F(x):
    # system = A @ x - b
    # return system.T @ system
    return np.linalg.norm(A @ x - b)**2

def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

def gradF(x):
    return 2 * A.T @ (A @ x - b)

def gradF2(x, delta2):
    return gradF(x) + 2 * delta2 * x

# Generación de datos
# np.random.seed(69 + 420 + int(9/11) + 80085)  # Para reproducibilidad
np.random.seed(0)
n, d = 5, 100
A = np.random.randn(n, d)
b = np.random.randn(n)
x0 = np.random.randn(d)

iterations = 1000

# Descomposición en valores singulares para obtener λ_max y σ_max
sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
sigma_max = np.max(sigma)
lambda_max = sigma_max ** 2 # https://en.wikipedia.org/wiki/Singular_value_decomposition, Relation to eigenvalue decomposition

# Paso de aprendizaje
s = 1 / lambda_max
delta2 = 1e-5 * sigma_max

s2 = 1 / (2*lambda_max + 2 * delta2) # Para regularización uso el valor de 2*λ_max + 2 * δ^2. ver https://chatgpt.com/share/d769e0c4-ec21-4396-be64-032f18cf483d


def gradient_descent(x0, iterations, s, regulartization=False, delta2 = 0):
    x = x0.copy()
    history = []
    for i in range(iterations):
        if not regulartization: 
            x -= s * gradF(x) 
            history.append(F(x))
        else:
            x -= s * gradF2(x, delta2)
            history.append(F2(x, delta2))
            
    return x, history

x_reg, history_reg = gradient_descent(x0, iterations, s2, regulartization=True, delta2=delta2)
x, history = gradient_descent(x0, iterations, s)

# Calculo la solución con SVD
x_svd = np.linalg.pinv(A) @ b

thickness = 3 # Esto en 50 es buenísimo

# Grafico
plt.figure()
plt.plot(history, label='$F(x)$', linewidth=thickness)
plt.plot(history_reg, label='$F(x) + \delta^2 {||x||}_2^2$', linewidth=thickness, url='https://youtu.be/dQw4w9WgXcQ') # SACAR LA URL
plt.xlabel('Iteraciones', fontsize=15)
plt.ylabel('Valor de la función de costo', fontsize=15)
plt.legend(fontsize=14)  # Increase the font size to make the legend box bigger
plt.title('Evolución de la función de costo utilizando el método de gradiente descendiente', fontsize=20)
# plt.ylim(0, 10)
plt.grid()
plt.show()

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
A = np.random.rand(n, d)
b = np.random.rand(n)
x_inicial = np.random.rand(d)
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