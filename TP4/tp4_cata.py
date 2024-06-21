import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

# Definición de funciones de costo y gradientes
def F(x, _ = 0):
    return np.linalg.norm(A @ x - b)**2

def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

def gradF(x):
    return 2 * A.T @ (A @ x - b)

def gradF2(x, delta2):
    return gradF(x) + 2 * delta2 * x

np.random.seed(0) #0, 6
n, d = 5, 100

A = np.random.rand(n, d)
b = np.random.rand(n)
x0 = np.random.rand(d)
# x0 = np.random.uniform(0, 1, d)

iterations = 1000

sigma = np.linalg.svd(A, full_matrices=False, compute_uv=False)
sigma_max = np.max(sigma)

# HF1(x) = 2 * A.T @ A
lambda_max = 2 * sigma_max**2 # multiplico por 2 porque el hessiano es 2*(A.T @ A) y sigma es la raiz cuadrada de los autovalores de A.T @ A

delta_constants = [0.001, 0.01, 0.1, 1, 10]  # Constante de regularización

delta2 = 1e-2 * sigma_max

# Paso de aprendizaje
step = 1 / lambda_max


def gradient_descent(x0, iterations, step, regularization=False, delta2 = 0):
    x = x0.copy()
    history = []
    history_unprocessed = [x.copy()]
    for i in range(iterations):
        if not regularization:
            x -= step * gradF(x)
            history.append(F(x))
            history_unprocessed.append(x.copy())
        else:
            x -= step * gradF2(x, delta2)
            history.append(F2(x, delta2))
            history_unprocessed.append(x.copy())

    return x, history, history_unprocessed

# Calculo la solución con SVD
x_svd = np.linalg.pinv(A) @ b

thickness = 2.5

def plotF1():
    # Grafico
    delta = 0.01
    delta2 = delta * sigma_max
    plt.figure()
    _, history_f1, _ = gradient_descent(x0, iterations, step)
    _, history_f2, _ = gradient_descent(x0, iterations, step, regularization=True, delta2=delta2)

    plt.plot(history_f1, linewidth=1.7, label="$F(x)$", color = "cadetblue")
    plt.plot(history_f2, linewidth=thickness, label="$F_2(x)$ con $\delta_2 =$ $10^{-2}$ $\cdot \sigma_{max}$", color= "lightcoral")
    plt.hlines(F(x_svd), 0, iterations, colors='darkslateblue', linestyles='dashed', label='$F(x)$ de la solución con SVD', linewidth=thickness)
   
    plt.xlabel('Iteraciones', fontsize=15)
    plt.ylabel('Valor de las funciones (en escala logarítmica)', fontsize=15)
    plt.legend(fontsize=14)  # Increase the font size to make the legend box bigger
    plt.yscale('log')
    plt.title('Evolución de $F(x)$ y $F_2(x)$ por iteración', fontsize=20)
    plt.grid(False)
    plt.show()

def plotF2():
    # Grafico
    plt.figure()
    colors = ['lightcoral', 'peachpuff', 'seagreen', 'cadetblue', 'midnightblue']
    
    for i, const in enumerate(delta_constants):
        delta2 = const * sigma_max
        _, history_f, _ = gradient_descent(x0, iterations, step, regularization=True, delta2=delta2)
        plt.plot(history_f, linewidth=thickness, label=f"$F_2(x)$ con $\delta^2 = {const} \cdot \sigma_{{max}}$", color=colors[i])
        
    plt.xlabel('Iteraciones', fontsize=15)
    plt.ylabel('Valor de $F_2(x)$', fontsize=15)
    plt.legend(fontsize=14)  # Increase the font size to make the legend box bigger
    plt.yscale('log')
    plt.title('Evolución de $F_2(x)$ por iteración', fontsize=20)
    plt.yticks(np.logspace(-2, 4, 7), fontsize=12)
    plt.grid()
    plt.show()

def plotNormOfX():
    x_svd = np.linalg.pinv(A) @ b
    
    colors = ['lightcoral', 'peachpuff', 'seagreen', 'cadetblue', 'midnightblue']
    
    plt.figure()
    plt.title('Norma de x por iteración', fontsize=20)
    _, _ , history_x1= gradient_descent(x0, iterations, step)
    plt.plot([np.linalg.norm(x) for x in history_x1], linewidth=thickness, label="$||x||_2$ con $F(x)$")

    for i, const in enumerate(delta_constants):
        delta2 = const * sigma_max
        _, _, history_x2 = gradient_descent(x0, iterations, step, regularization=True, delta2=delta2)
        plt.plot([np.linalg.norm(x) for x in history_x2], linewidth=thickness, label=f"$||x||_2$ con $F_2(x)$ y $\delta^2$ = {const} $\cdot \sigma_{{max}}$", color = colors[i])
    plt.xlabel('Iteraciones', fontsize=15)
    plt.ylabel('Valor de la norma de x', fontsize=15)
    


    plt.hlines(np.linalg.norm(x_svd), 0, iterations, colors='teal', linestyles='dashed', label='$||x||_2$ de la solución SVD', linewidth=thickness)

    plt.legend(fontsize=12, loc='center right')  # Increase the font size to make the legend box bigger
    plt.yscale('log')
    plt.grid()
    plt.show()

def showRelativeErrors():
    colors = ['lightcoral', 'wheat', 'darkseagreen', 'cadetblue', 'steelblue', 'midnightblue']
    
    def calc_error(x, x_svd, function=F2, delta=0):
        return np.abs(function(x, delta) - function(x_svd, delta))

    x_svd = np.linalg.pinv(A) @ b
    plt.figure()
    plt.title('Error absoluto de $||A \cdot x - b||_2$ luego de 1000 iteraciones', fontsize=20)
    x_f1, _ , _= gradient_descent(x0, iterations, step)
    error = calc_error(x_f1, x_svd, F)

    tick_labels = ['$F(x)$', '$F_2(x), \delta^2 = 0.001$', '$F_2(x), \delta^2 = 0.01$', '$F_2(x), \delta^2 = 0.1$', '$F_2(x), \delta^2 = 1$', '$F_2(x), \delta^2 = 10$']
    plt.bar(0, error, label="$F(x)$", color = colors[0])
    for i in range(len(delta_constants)):
        delta = delta_constants[i] * sigma_max
        x_i, _, _ = gradient_descent(x0, iterations, step, regularization=True, delta2=delta * sigma_max)
        error = calc_error(x_i, x_svd, F2, delta)
        print(f"Error relativo para δ^2 = {delta}: {error}")
        plt.bar(i+1, error, label=f"$\delta^2 = {delta_constants[i]}$$\cdot \sigma_{{max}}$", color = colors[i+1])
    # plt.xlabel('$\delta^2$', fontsize=15)
    plt.ylabel('Error absoluto de $||A \cdot x - b||_2$', fontsize=15)
    plt.legend(fontsize=14)  # Increase the font size to make the legend box bigger
    plt.yscale('log')
    plt.xticks(range(len(delta_constants) + 1), tick_labels, fontsize=12)
    # plt.grid()
    plt.show()

def pca_transform(data, n_components=2):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca

def plot_isocost_contour_and_path():
    # Realiza el gradiente descendente y almacena la historia de x
    _, _, history_x_f1 = gradient_descent(x0, iterations, step)

    delta = 0.01
    delta2 = delta * sigma_max
    _, _, history_x_f2 = gradient_descent(x0, iterations, step, regularization=True, delta2=delta2)
    _, _, history_x_f2_2 = gradient_descent(x0, iterations, step, regularization=True, delta2=1 * sigma_max)
    _, _, history_x_f2_3 = gradient_descent(x0, iterations, step, regularization=True, delta2=2 * sigma_max)

    # Convierte la historia de x a un arreglo numpy para PCA
    history_x_f1 = np.array(history_x_f1)
    history_x_f2 = np.array(history_x_f2)
    history_x_f2_2 = np.array(history_x_f2_2)


    # Realiza PCA para reducir dimensionalidad a 2D
    history_transformed_f1, pca_f1 = pca_transform(history_x_f1)
    history_transformed_f2 = pca_f1.transform(history_x_f2)
    history_transformed_f2_2 = pca_f1.transform(history_x_f2_2)
    history_transformed_f2_3 = pca_f1.transform(history_x_f2_3)
    svd_solution = pca_f1.transform([x_svd])[0]

    # Genera puntos de una malla para graficar el contorno de los costos
    x_min, x_max = history_transformed_f1[:, 0].min() - 1, history_transformed_f1[:, 0].max() + 1
    y_min, y_max = history_transformed_f1[:, 1].min() - 1, history_transformed_f1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calcula el costo en cada punto de la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original_space = pca_f1.inverse_transform(mesh_points)
    zz = np.array([F(point) for point in mesh_points_original_space])
    zz = zz.reshape(xx.shape)

    # Grafica el contorno de los costos
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.15)
    plt.contour(xx, yy, zz, levels=50, cmap='viridis')
    plt.plot(history_transformed_f1[:, 0], history_transformed_f1[:, 1], 'r-o', markersize=3, linewidth=2, label='Trayectoria GD con $F(x)$')
    plt.scatter(svd_solution[0], svd_solution[1], color='teal', label='Solución SVD', s=100)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Camino del Gradiente Descendiente $F(x)$ en el Contorno de Isocostos')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.15)
    plt.contour(xx, yy, zz, levels=50, cmap='viridis')
    plt.plot(history_transformed_f2[:, 0], history_transformed_f2[:, 1], 'g-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = {delta}$')
    plt.plot(history_transformed_f2_2[:, 0], history_transformed_f2_2[:, 1], 'b-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = 1$')
    plt.plot(history_transformed_f2_3[:, 0], history_transformed_f2_3[:, 1], 'm-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = 5$')
    plt.scatter(svd_solution[0], svd_solution[1], color='teal', label='Solución SVD', s=100)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Camino del Gradiente Descendiente $F_2(x)$ en el Contorno de Isocostos')
    plt.legend()

    plt.suptitle('Camino del Gradiente Descendiente en el Contorno de Isocostos', fontsize=20)
    plt.show()

    # plt.figure()
    # plt.contour(xx, yy, zz, levels=50, cmap='viridis')
    # plt.plot(history_transformed_f1[:, 0], history_transformed_f1[:, 1], 'r-o', markersize=3, linewidth=2, label='Trayectoria GD con $F(x)$')
    # plt.plot(history_transformed_f2[:, 0], history_transformed_f2[:, 1], 'b-o', markersize=3, linewidth=2, label=f'Trayectoria GD con $F_2(x)$ y $\delta_2 = {delta}$')
    # plt.scatter(svd_solution[0], svd_solution[1], color='teal', label='Solución SVD', s=100)
    # plt.xlabel('Componente Principal 1')
    # plt.ylabel('Componente Principal 2')
    # plt.title('Camino del Gradiente Descendiente en el Contorno de Isocostos')
    # plt.legend()
    # plt.show()

def plot_isocost_contour_and_path_3d():
    # Realiza el gradiente descendente y almacena la historia de x
    _, _, history = gradient_descent(x0, iterations, step)

    # Convierte la historia de x a un arreglo numpy para PCA
    history_array = np.array(history)

    # Realiza PCA para reducir dimensionalidad a 2D
    history_transformed, pca = pca_transform(history_array)

    # Genera puntos de una malla para graficar el contorno de los costos
    side_size = 2.5
    x_min, x_max = -side_size, side_size
    y_min, y_max = -side_size, side_size
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calcula el costo en cada punto de la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original_space = pca.inverse_transform(mesh_points)
    zz = np.array([F(point) for point in mesh_points_original_space])
    zz = zz.reshape(xx.shape)

    # Crea el gráfico 3D
    fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy)])

    # Añade el camino del gradiente descendente como una línea 3D
    fig.add_trace(
        go.Scatter3d(
            x=history_transformed[:, 0],
            y=history_transformed[:, 1],
            z=[F(point) for point in history_array],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="red", width=10),  # Line width increased to 4
            name="Solución"  # Label for the line
        )
    )
    fig.update_layout(
        title="Camino del Gradiente Descendente en el Contorno de Isocostos 3D",
        autosize=True,
        scene=dict(
            xaxis_title="Componente Principal 1",
            yaxis_title="Componente Principal 2",
            zaxis_title="Costo",
        ),
    )

    # Plot the SVD solution as a Green point

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[F(x_svd)],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="Solución SVD"
        )
    )


    fig.add_annotation(
        text="Solución",
        xref="paper", yref="paper",
        x=0.05, y=0.95, showarrow=False,
        font=dict(size=14, color="red")
    )

    pyo.plot(fig)

def compareSteps():
    steps = [1/10, 21/10]
    plt.figure()
    plt.title('Evolución de $F(x)$ y $F_2(x)$ por iteración cambiando el paso', fontsize=20)
    
    plt.hlines(F(x_svd), 0, iterations, colors='teal', linestyles='dashed', label='$F(x)$ de la solución SVD', linewidth=thickness)
    
    # colors = ['lightcoral', 'peachpuff', 'seagreen', 'cadetblue', 'midnightblue']
    colors = ['lightcoral', 'wheat', 'darkseagreen', 'cadetblue', 'steelblue', 'midnightblue']
    
    for i, step_i in enumerate(steps):
        step_f = step_i / lambda_max
        label = f"step = {step_i}" + "$\cdot \lambda_{max}^{-1}$"
        # label = "test"
        _, history_f1, _ = gradient_descent(x0, iterations, step_f)
        plt.plot(history_f1, linewidth=thickness, label=("$F(x)$ con " + label), color = colors[i])
        _, history_f2, _ = gradient_descent(x0, iterations, step_f, regularization=True, delta2=0.01 * sigma_max)
        plt.plot(history_f2, linewidth=thickness, label=("$F_2(x)$ con $\delta_2$ = $0.01 \cdot \sigma_{max}$ y" + label), color = colors[i+1])

    plt.xlabel('Iteraciones', fontsize=15)
    plt.ylabel('Valor de $F(x)$ y $F_2(x)$', fontsize=15)
    plt.legend(fontsize=14)  # Increase the font size to make the legend box bigger
    plt.yscale('log')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    plotF1()
    plotF2()
    showRelativeErrors()
    plotNormOfX()
    plot_isocost_contour_and_path()
    compareSteps()

    # plot_isocost_contour_and_path_2d()
    plot_isocost_contour_and_path_3d()