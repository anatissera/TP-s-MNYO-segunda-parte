import numpy as np
import matplotlib.pyplot as plt

# Configuraciones iniciales
n = 5  # Dimensión del problema
d = 100  # Dimensión del espacio de parámetros
iterations = 1000
delta2 = 1e-2

# Generación de matrices y vectores aleatorios
np.random.seed(0)  # Para reproducibilidad
A = np.random.randn(n, d)
b = np.random.randn(n)

# Función de costo sin regularización
def F(x):
    return np.linalg.norm(A @ x - b)**2

# Gradiente de la función de costo sin regularización
def gradF(x):
    return 2 * A.T @ (A @ x - b)

# Función de costo con regularización L2
def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

# Gradiente de la función de costo con regularización L2
def gradF2(x, delta2):
    return gradF(x) + 2 * delta2 * x

# Inicialización aleatoria
x0 = np.random.randn(d)

# Descomposición en valores singulares para obtener λ_max y σ_max
U, sigma, VT = np.linalg.svd(A, full_matrices=False)
sigma_max = np.max(sigma)
lambda_max = sigma_max**2

# Paso de aprendizaje sin regularización
s = 1 / lambda_max

# Paso de aprendizaje con regularización
lambda_max_reg = 2 * lambda_max + 2 * delta2
s2 = 1 / lambda_max_reg

# Gradiente descendente sin regularización
x_gd = x0.copy()
history_F = [x_gd.copy()]

for i in range(iterations):
    x_gd -= s * gradF(x_gd)
    history_F.append(x_gd.copy())

# Gradiente descendente con regularización L2
x_gd_reg = x0.copy()
history_F2 = [x_gd_reg.copy()]

for i in range(iterations):
    x_gd_reg -= s2 * gradF2(x_gd_reg, delta2)
    history_F2.append(x_gd_reg.copy())

history_F = np.array(history_F)
history_F2 = np.array(history_F2)

# Reducir las dimensiones de history_F y history_F2 a 2D para visualización
history_F_reduced = history_F[:, :2] # sin pca no se puede ver en 2d
history_F2_reduced = history_F2[:, :2]

# Generar una malla para las curvas de isocosto porque no uso PCA
x_range = np.linspace(-4, 4, 400)
y_range = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)
Z2 = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.zeros(d)
        x[:2] = [X[i, j], Y[i, j]]
        Z[i, j] = F(x)
        Z2[i, j] = F2(x, delta2)

# Graficar las curvas de isocosto y el historial del gradiente descendente
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(history_F_reduced[:, 0], history_F_reduced[:, 1], 'o-', color='orangered', label='Trayectoria GD')
plt.title('Curvas de Isocosto (Sin Regularización)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.subplot(1, 2, 2)
plt.contour(X, Y, Z2, levels=50, cmap='viridis')
plt.plot(history_F2_reduced[:, 0], history_F2_reduced[:, 1], 'o-', color='orangered', label='Trayectoria GD Regularizado')
plt.title('Curvas de Isocosto (Con Regularización)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.tight_layout()
plt.show()