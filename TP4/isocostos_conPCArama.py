import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configurations
n = 2  # Problem dimension
d = 2  # Parameter space dimension
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

# Learning step without regularization
s = 1 / lambda_max

# Learning step with regularization
lambda_max_reg = 2 * lambda_max + 2 * delta2
s2 = 1 / lambda_max_reg

# Gradient descent without regularization
x_gd = x0.copy()
history_F = [x_gd.copy()]

for i in range(iterations):
    x_gd -= s * gradF(x_gd)
    history_F.append(x_gd.copy())

# Gradient descent with L2 regularization
x_gd_reg = x0.copy()
history_F2 = [x_gd_reg.copy()]

for i in range(iterations):
    x_gd_reg -= s2 * gradF2(x_gd_reg, delta2)
    # x_gd_reg -= s * gradF2(x_gd_reg, delta2)
    history_F2.append(x_gd_reg.copy())

# Convert history to NumPy arrays for easier plotting
history_F = np.array(history_F)
history_F2 = np.array(history_F2)

# Fit PCA on the combined trajectories
all_points = np.vstack((history_F, history_F2))  # Combine all points
pca = PCA(n_components=2)
pca.fit(all_points)

# Transform the trajectories using PCA
history_F_pca = pca.transform(history_F)
history_F2_pca = pca.transform(history_F2)

# Adjust mesh grid range to cover a broader area
x_range = np.ptp(history_F_pca[:, 0])  # Range of x in PCA-transformed space
y_range = np.ptp(history_F_pca[:, 1])  # Range of y in PCA-transformed space
x_min_pca, x_max_pca = -1, 1
y_min_pca, y_max_pca = -1, 1
# Generate a new mesh grid in the PCA-transformed space
xx_pca, yy_pca = np.meshgrid(np.linspace(x_min_pca, x_max_pca, 100), np.linspace(y_min_pca, y_max_pca, 100))

# Transform back the mesh grid points to the original space for iso-cost computation
mesh_points_pca_inverse = pca.inverse_transform(np.c_[xx_pca.ravel(), yy_pca.ravel()])

# Compute iso-cost values for the transformed back mesh grid
Z_pca = np.array([F(np.array([x, y])) for x, y in mesh_points_pca_inverse])
Z_pca = Z_pca.reshape(xx_pca.shape)

levels = 25  # Number of iso-cost curves

# Plotting the transformed trajectories and iso-cost curves with a colorful cmap, without filling between curves
plt.figure(figsize=(14, 6))

# Without regularization
plt.subplot(1, 2, 1)
plt.contour(xx_pca, yy_pca, Z_pca, levels=levels, cmap='viridis')
plt.plot(history_F_pca[:, 0], history_F_pca[:, 1], 'o-', color='orangered', label='Trayectoria GD')
# plt.title('PCA Transformed Trajectory (Without Regularization)')
plt.title('Curvas de Isocosto con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()

# With regularization
plt.subplot(1, 2, 2)
plt.contour(xx_pca, yy_pca, Z_pca, levels=levels, cmap='viridis')
plt.plot(history_F2_pca[:, 0], history_F2_pca[:, 1], 'o-', color='orangered', label='Trayectoria GD Regularizado')
# plt.title('PCA Transformed Trajectory (With Regularization)')
plt.title('Curvas de Isocosto con regularización con PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()

plt.tight_layout()
plt.show()