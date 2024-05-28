import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels = np.loadtxt('tp3/y.txt')

def processMatrix(matrix):
    matrix = pd.read_csv(path, header=None)
    # Elimino la primera fila y columna que son los nombres de las columnas y filas
    matrix = matrix.drop(0, axis=0)
    matrix = matrix.drop(0, axis=1)

    # Convierto la matriz a un array de numpy
    matrix = matrix.to_numpy()
    
    return matrix

# def similarity_matrix(matrix, deviation):
#     n = matrix.shape[0]
#     sim_matrix = np.zeros((n, n))
#     for i in range(n):
#         if i%50 == 0: print(f"cambie de columna, estoy en {i}")
#         for j in range(n):
#             sim_matrix[i, j] = np.exp(-(np.linalg.norm(matrix[i] - matrix[j]))/2*deviation**2)
#     return sim_matrix

def similarity_matrix(matrix, deviation):
    # matrix = matrix - np.mean(matrix, axis=0)
    diff = matrix[:, None] - matrix
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    sim_matrix = np.exp(-np.sqrt(dist_sq) / (2 * deviation**2))
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
    # hace psa 
    
    v_transpose = v[:dimension, :].T
    print(f"Dimensiones de V transpuesta: {v_transpose.shape}")
    print(f"Dimensiones de Matriz Original: {matrix.shape}")
    matrix_svd = matrix @ v_transpose
    print(f"Dimensiones de MatrizSVD: {matrix_svd.shape}")
    
    sim_matrix = similarity_matrix(matrix_svd, deviation)
    print(f"Dimensiones de MatrizSimilitudSVD: {sim_matrix.shape}")

    
    plt.figure()
    plt.imshow(sim_matrix)
    plt.colorbar()
    plt.title(f"Matriz de similaritud con desviación {deviation} y reducción de dimensiones a {dimension}")
    plt.show()
    
    if dimension == 2:
        plt.scatter(matrix_svd[:, 0], matrix_svd[:, 1], marker='o', color='teal') # c=labels, cmap= 'vidris'
        plt.title(f"Reducción de dimensiones a {dimension}")
        plt.show()
        return
    elif dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(matrix_svd[:, 0], matrix_svd[:, 1], matrix_svd[:, 2], c=labels, marker='o')
        # fig.colorbar(scatter, ax=ax)
        plt.title(f"Reducción de dimensiones a {dimension}")
        plt.show()
        return
    else:
        return
    
if __name__ == '__main__':
    path = 'TP3/dataset02.csv'
    deviation = 1
    iterate = True
    
    matrix = processMatrix(path)
    
    # if iterate:
    #     deviation = 100    
    #     for i in range(5):
    #         print(f"Desviación: {deviation}")
    #         show_similarity_matrix(matrix, deviation)
    #         deviation = deviation/10
    # else:     
    #     show_similarity_matrix(matrix, deviation)
    
    reduceDimensions(matrix, deviation, 2) #grafico ocn dimensión 2
    
    