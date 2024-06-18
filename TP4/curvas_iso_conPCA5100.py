import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configurations
n = 5  # Problem dimension
d = 100  # Parameter space dimension
iterations = 100
delta2 = 1e-2

# Random matrices and vectors generation
np.random.seed(0)  # For reproducibility
A = np.random.randn(n, d)
b = np.random.randn(n)

# Cost function without regularization
def F(x):
    return np.linalg.norm(A @ x - b)**2

# Gradient of the cost function without regularization
def gradF(x):
    return 2 * A.T @ (A @ x - b)

# Cost function with L2 regularization
def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

# Gradient of the cost function with L2 regularization
def gradF2(x, delta2):
    return gradF(x) + 2 * delta2 * x

# Random initialization
x0 = np.random.randn(d)

# Singular value decomposition to obtain λ_max and σ_max
U, sigma, VT = np.linalg.svd(A, full_matrices=False)
sigma_max = np.max(sigma)
lambda_max = sigma_max**2

# Learning step sin regularización
s = 1 / lambda_max

# Learning step con regularización L2
lambda_max_reg = 2 * lambda_max + 2 * delta2
s2 = 1 / lambda_max_reg

# Gradient descent sin regularización
x_gd = x0.copy()
history_F = [x_gd.copy()]

for i in range(iterations):
    x_gd -= s * gradF(x_gd)
    history_F.append(x_gd.copy())

# Gradient descent con L2 regularización
x_gd_reg = x0.copy()
history_F2 = [x_gd_reg.copy()]

for i in range(iterations):
    x_gd_reg -= s2 * gradF2(x_gd_reg, delta2)
    history_F2.append(x_gd_reg.copy())

history_F = np.array(history_F)
history_F2 = np.array(history_F2)

# Fit PCA
all_points = np.vstack((history_F, history_F2))  # Combinar todos los puntos
pca = PCA(n_components=2)
pca.fit(all_points)

# Transformar trayectorias a espacio PCA
history_F_pca = pca.transform(history_F)
history_F2_pca = pca.transform(history_F2)

# Rango de x e y en el espacio transformado por PCA
x_min_pca, x_max_pca = history_F_pca[:, 0].min(), history_F_pca[:, 0].max()
y_min_pca, y_max_pca = history_F_pca[:, 1].min(), history_F_pca[:, 1].max()

# Generar una nueva malla en el espacio transformado por PCA
xx_pca, yy_pca = np.meshgrid(np.linspace(x_min_pca-1, x_max_pca+1, 100), np.linspace(y_min_pca-1, y_max_pca+1, 100))

# Transformar la malla al espacio original
mesh_points_pca_inverse = pca.inverse_transform(np.c_[xx_pca.ravel(), yy_pca.ravel()])

# Computar el iso-costo en el espacio original
Z_pca = np.array([F(np.array(point)) for point in mesh_points_pca_inverse])
Z_pca = Z_pca.reshape(xx_pca.shape)

levels = 25  # Número de curvas de isocosto

plt.figure(figsize=(14, 6))

# Sin regularización
plt.subplot(1, 2, 1)
plt.contour(xx_pca, yy_pca, Z_pca, levels=levels, cmap='viridis')
plt.plot(history_F_pca[:, 0], history_F_pca[:, 1], 'o-', color='orangered', label='Trayectoria GD')
plt.title('Curvas de Isocosto con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()

# Con regularización
plt.subplot(1, 2, 2)
plt.contour(xx_pca, yy_pca, Z_pca, levels=levels, cmap='viridis')
plt.plot(history_F2_pca[:, 0], history_F2_pca[:, 1], 'o-', color='orangered', label='Trayectoria GD Regularizado')
plt.title('Curvas de Isocosto con regularización con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()

plt.tight_layout()
plt.show()