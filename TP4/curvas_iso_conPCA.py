import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n = 2  
d = 2 
iterations = 1000
delta2 = 1e-2

np.random.seed(0)  
A = np.random.randn(n, d)
b = np.random.randn(n)

def F(x):
    return np.linalg.norm(A @ x - b)**2

def gradF(x):
    return 2 * A.T @ (A @ x - b)

def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

def gradF2(x, delta2):
    return gradF(x) + 2 * delta2 * x


x0 = np.random.randn(d)

U, sigma, VT = np.linalg.svd(A, full_matrices=False)
sigma_max = np.max(sigma)
lambda_max = sigma_max**2

# Learning step sin regularización
s = 1 / lambda_max

# Learning step con regularización L2
lambda_max_reg = 2 * lambda_max + 2 * delta2
s2 = 1 / lambda_max_reg
# con delta2 acomodo la magnitud, es una penalización

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

history_F_pca = pca.transform(history_F)
history_F2_pca = pca.transform(history_F2)

x_range = np.ptp(history_F_pca[:, 0])  
y_range = np.ptp(history_F_pca[:, 1])  
x_min_pca, x_max_pca = -1, 1
y_min_pca, y_max_pca = -1, 1

# Generar una malla
xx_pca, yy_pca = np.meshgrid(np.linspace(x_min_pca, x_max_pca, 1000), np.linspace(y_min_pca, y_max_pca, 1000))

# transformar el mesh grid a espacio original
mesh_points_pca_inverse = pca.inverse_transform(np.c_[xx_pca.ravel(), yy_pca.ravel()])

# computar el iso-costo en el espacio original
Z_pca = np.array([F(np.array(points)) for points in mesh_points_pca_inverse])
Z_pca = Z_pca.reshape(xx_pca.shape)

levels = 25  # número de curvas de isocosto


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