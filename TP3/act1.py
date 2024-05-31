import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


labels = np.loadtxt('tp3/y.txt')

path = "TP3/dataset02.csv"

def processMatrix(matrix):
    matrix = pd.read_csv(path, header=None)
    # Elimino la primera fila y columna que son los nombres de las columnas y filas
    matrix = matrix.drop(0, axis=0)
    matrix = matrix.drop(0, axis=1)

    # Convierto la matriz a un array de numpy
    matrix = matrix.to_numpy()
    matrix = matrix - np.mean(matrix, axis=0) # esta cosa centra la matriz

    return matrix

# def similarity_matrix(matrix, deviation):
#     n = matrix.shape[0]
#     sim_matrix = np.zeros((n, n))
#     for i in range(n):
#         if i%50 == 0: print(f"cambie de columna, estoy en {i}")
#         for j in range(n):
#             sim_matrix[i, j] = np.exp(-(np.linalg.norm(matrix[i] - matrix[j]))/2*deviation**2)
#     return sim_matrix

def euclidean_distances(X):
    # Calculamos la matriz de productos internos
    XXT = X @ X.T
    # Extraemos la diagonal (normas cuadradas de cada punto)
    norms = np.diag(XXT)
    # Calculamos la matriz de distancias euclidianas usando broadcasting
    distances = np.sqrt(norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * XXT)
    return distances

def similarity_matrix(matrix, deviation):
    # diff = matrix[:, None] - matrix
    # dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    # sim_matrix = np.exp(-np.sqrt(dist_sq) / (2 * deviation**2))
    sim_matrix = np.exp(-euclidean_distances(matrix) / (2 * deviation**2))
    return sim_matrix


def show_similarity_matrix(matrix, deviation):

    sim_matrix = similarity_matrix(matrix, deviation)

    plt.figure()
    plt.imshow(sim_matrix)
    plt.colorbar()
    plt.title(f"Matriz de similaritud con desviación {deviation}")
    plt.show()
    
def reduceDimensions(matrix, deviation, dimension):
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    print(f"Dimensiones de U: {u.shape}")
    print(f"Dimensiones de S: {s.shape}")
    print(f"Dimensiones de V: {v.shape}")
    print()    

    # Reducimos la dimensión de la matriz
    U_reduced = u[:, :dimension]
    S_reduced = np.diag(s[:dimension])
    matrix_svd = U_reduced @ S_reduced
    print(f"Dimensiones de MatrizSVD: {matrix_svd.shape}")
    
    sim_matrix = similarity_matrix(matrix_svd, deviation)
    print(f"Dimensiones de MatrizSimilitudSVD: {sim_matrix.shape}")
    
    plt.figure()
    plt.imshow(sim_matrix)
    plt.colorbar()
    plt.title(f"Matriz de similaritud con desviación {deviation} y reducción de dimensiones a {dimension}")
    plt.show()
    
    if dimension == 2:
        
        plt.figure()
        plt.imshow(u)
        plt.colorbar()
        plt.title("Matriz U")
        plt.show()
        
        plt.figure()
        plt.bar(range(len(s)), s)
        plt.title("Valores singulares")
        plt.xlabel("Número de valor singular")
        plt.ylabel("Varianza del valor singular")
        plt.show()
        
        plt.figure()
        plt.imshow(v)
        plt.colorbar()
        plt.title("Matriz V")
        plt.show()
        
        plt.scatter(matrix_svd[:, 0], matrix_svd[:, 1], c=labels, cmap='ocean', marker='o')
        plt.title(f"Reducción de dimensiones a {dimension}")
        plt.show()
        return
    elif dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(matrix_svd[:, 0], matrix_svd[:, 1], matrix_svd[:, 2], cmap='ocean', c=labels, marker='o')
        plt.title(f"Reducción de dimensiones a {dimension}")
        plt.show()
        return
    else:
        return
    
def iterateDimensions(matrix, deviation, dimensions):
        dimension = dimensions[0] 
        for i in range(1,5):
            print(f"Desviación: {deviation}")
            reduceDimensions(matrix, deviation, dimension)
            dimension = dimensions[i]  
    
def iterateDeviation(matrix):
    deviation = 10
    for i in range(5):
        print(f"Desviación: {deviation}")
        show_similarity_matrix(matrix, deviation)
        deviation = deviation/10

def leastSquares(matrix, y):
    # Realizamos la descomposición SVD
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    errors = []
    dimensions = []

    for d in range(1, matrix.shape[1] + 1):
        # Realizamos la reducción de dimensiones
        U_reduced = u[:, :d]
        S_reduced = np.diag(s[:d])
        V_reduced = v[:d, :]
        matrix_svd = U_reduced @ S_reduced @ V_reduced
        
        # Calculamos la matriz de coeficientes
        A = np.linalg.inv(S_reduced) @ U_reduced.T
        # A = np.linalg.pinv(U_reduced @ S_reduced)
        
        # Calculamos los coeficientes
        beta = A @ y
        
        # Calculamos el error
        error = np.linalg.norm(matrix @ V_reduced.T @ beta - y) / np.linalg.norm(y)

        errors.append(error)
        dimensions.append(d)
    
    best_dimension = dimensions[np.argmin(errors)]    
            
    plt.figure()
    plt.plot(dimensions, errors)
    plt.title("Error relativo entre normas en función de la dimensión")
    plt.xlabel("Dimensión")
    plt.ylabel("% de Error")
    # plt.ylim(0.75, 0.80)
    plt.show()
    
    print(f"La mejor dimensión es {best_dimension} con un error de {errors[best_dimension-1]}")
        
if __name__ == '__main__':
    deviation = 1
    dimensions = [2, 3, 6, 10, 20]
    iterate = False
    
    matrix = processMatrix(path)
    
    if iterate:
        # iterateDimensions(matrix, deviation, dimensions)
        iterateDeviation(matrix)
    else:
        # reduceDimensions(matrix, deviation, 2)
        leastSquares(matrix, labels)