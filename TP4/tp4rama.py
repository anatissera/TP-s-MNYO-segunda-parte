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