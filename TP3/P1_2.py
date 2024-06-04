# De las p dimensiones originales del dataset, cuales son las mas representativas con respecto a las
# dimensiones d obtenidas por SVD? Indicar que dimensiones originales del conjunto p son las mas
# importantes y el método utilizado para determinarlas.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import euclidean_distances

from P1_1 import load_data, normalize_dataset, normalize_dataset_martin


def plot_singular_values(X, indices=[3, 7, 11]):
 
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    S = S[:102]
    
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Valores singulares $\sigma_i$', color = "darkcyan", linewidth=2)

    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot(idx, S[idx], 'o', color=color, markersize = 4) 
        plt.vlines(x=idx, ymin=S[-1], ymax=S[idx], color=color, linestyle=style, label=f'd={(idx)}')
        plt.hlines(y=S[idx], xmin=0, xmax=idx, color=color, linestyle=style)
    
    print (f"Valores singulares para d = 2: {S[1]:.4f}")
    print (f"Valores singulares para d = 6: {S[5]:.4f}")
    print (f"Valores singulares para d = 10: {S[9]:.4f}")
    print (f"Valores singulares para d = 102: {S[101]:.4f}")
    print (f"Valores singulares para d = 1: {S[0]:.4f}")
    
    print (f"Diferencia d = 10 - d = 2: {S[9] - S[1]:.4f}")
    
    
    plt.yscale('log')
    plt.xlabel('$i$', fontsize=17)
    plt.ylabel('Valores singulares $\sigma_i$', fontsize=17)
    plt.title('Figura de los valores singulares del dataset $X \{\sigma_i\}_{i=1}^{102}$', fontsize=18)
    plt.legend(fontsize = 14)
    plt.grid(False)
    plt.show()

def valores_singulares_acumulada(dataset):
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)
    
    S_squared = S**2
    proporcion_acumulada = np.cumsum(S_squared) / np.sum(S_squared)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(S) + 1), proporcion_acumulada, 'o-', markersize=3, color="darkcyan", linewidth=2)

    plt.plot([0, 1], [0, proporcion_acumulada[0]], 'o-', markersize=3, color="darkcyan", linewidth=2)
   
    indices = [2, 6, 10]
    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot([idx, idx], [0, proporcion_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5, label=f'd={idx}')
        plt.plot([0, idx], [proporcion_acumulada[idx-1], proporcion_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5)

    umbral_varianza = 0.5
    plt.plot([0, len(S)], [umbral_varianza, umbral_varianza], 'b--', linewidth=1.4, label='50% de varianza acumulada')

    plt.xlabel('$d$', fontsize=17)
    plt.ylabel(r'$\dfrac{\sum_{i=1}^{d} \sigma_i ^2}{\sum_{i=1}^{n} \sigma_i ^2}$', rotation=0, fontsize=17, labelpad=30, y=0.375)
    plt.title('Varianza según dimensión: Proporción de la suma acumulada de $\{\sigma_i\}$ de $A$', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.show()
    

    print(f"Varianza acumulada para d = 1: {proporcion_acumulada[0]:.4f}")
    print(f"Varianza acumulada para d = 2: {proporcion_acumulada[1]:.4f}")
    print(f"Varianza acumulada para d = 6: {proporcion_acumulada[5]:.4f}")
    print(f"Varianza acumulada para d = 10: {proporcion_acumulada[9]:.4f}")
    print(f"Diferencia de varianza acumulada entre d = 10 y d = 2: {(proporcion_acumulada[9] - proporcion_acumulada[1]):.4f}")
    print(f"Diferencia de varianza acumulada entre d = 6 y d = 2: {(proporcion_acumulada[5] - proporcion_acumulada[1]):.4f}")
    print(f"Diferencia de varianza acumulada entre d = 10 y d = 6: {(proporcion_acumulada[9] - proporcion_acumulada[5]):.4f}")


def media_acumulada_valores_singulares(dataset):
    U, S, Vt = np.linalg.svd(dataset, full_matrices=False)
    
    S_squared = S**2

    diferencias = np.diff(S_squared)
    
    acumulacion_diferencias = np.cumsum(np.abs(diferencias))
    
    media_acumulada = acumulacion_diferencias / np.arange(1, len(diferencias) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(media_acumulada) + 1), media_acumulada, 'o-', markersize=3, color="darkcyan", linewidth=2)
    
    indices = [2, 6, 10]
    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot([idx, idx], [0, media_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5, label=f'd={idx}')
        plt.plot([0, idx], [media_acumulada[idx-1], media_acumulada[idx-1]], color=color, linestyle=style, linewidth=1.5)
  

    plt.xlabel('$i$', fontsize=17)
    plt.ylabel('Media Acumulada de las Diferencias de los Valores Singulares', fontsize=17)
    plt.title('Media Acumulada de las Diferencias de los Valores Singulares de $A$', fontsize=18)
    plt.grid(False)
    plt.show()
    
    
def main():
    X, Y = load_data()
    dims = [2, 6, 10]
    
    # plot_singular_values(X, dims)
    plot_singular_values(normalize_dataset(X), dims)
    
    # valores_singulares_acumulada(X)
    valores_singulares_acumulada(normalize_dataset(X))
    
    # media_acumulada_valores_singulares(X)
    media_acumulada_valores_singulares(normalize_dataset(X))

    
if __name__ == "__main__":
    main()
    