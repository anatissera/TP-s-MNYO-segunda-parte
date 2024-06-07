# De las p dimensiones originales del dataset, cuales son las mas representativas con respecto a las
# dimensiones d obtenidas por SVD? Indicar que dimensiones originales del conjunto p son las mas
# importantes y el método utilizado para determinarlas.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from P1_1 import load_data, normalize_dataset, normalize_dataset_martin, similarity_matrix, pca_with_svd


def plot_singular_values(X, indices=[1, 5, 9]):
 
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    S = S[:102]
    
    plt.figure(figsize=(15, 5))
    plt.plot(range(1, len(S) +1 ), S, 'o-', label='Valores singulares $\sigma_i$', color="darkcyan", linewidth=1.5, markersize=2.4)

    styles = ['dotted', 'dashed', 'dashdot']
    colors = ['magenta', 'orange', 'grey']
    
    for idx, style, color in zip(indices, styles, colors):
        plt.plot(idx+1, S[idx], 'o', color=color, markersize=5) 
        plt.vlines(x=idx+1, ymin=0, ymax=S[idx], color=color, linestyle=style, label=f'd={idx+1}')
        plt.hlines(y=S[idx], xmin=0, xmax=idx+1, color=color, linestyle=style)
        
    print(f"Valores singulares para d = 1: {S[0]:.4f}")
    for idx in indices:
        print(f"Valores singulares para d = {idx}: {S[idx-1]:.4f}")
    print(f"Valores singulares para d = 102: {S[101]:.4f}")
    
    print(f"Diferencia d = 10 - d = 2: {S[9] - S[0]:.4f}")
    
    plt.yscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=25)
    plt.xlabel('$i$', fontsize=16)
    plt.ylabel('Valores singulares $\sigma_i$', fontsize=16)
    plt.title('Figura de los valores singulares del dataset $X \{\sigma_i\}_{i=1}^{102}$', fontsize=18)
    plt.legend(fontsize=15)
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
    
    X = normalize_dataset(X)
    
    plot_singular_values(X)

    valores_singulares_acumulada(X)

    media_acumulada_valores_singulares(X)
    
    _, _, _, vt = pca_with_svd(X, 2)
    correlation_matrix= similarity_matrix(X.T, 1)

    plt.figure()
    plt.bar(range(1, len(vt[0])+1), (vt[0, :]), color='darkcyan')
    plt.title("Primer Vector de $V^T$", fontsize = 16)
    plt.xlabel("Componente", fontsize = 14)
    plt.ylabel("Valor del Componente", fontsize = 14)
    # plt.xticks(range(1, len(vt[0])+1), [str(i+101) for i in range(len(vt[0]))])
    plt.grid()
    plt.show()
        
    plt.figure(figsize=(8, 8))
    plt.imshow(X, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Matriz original", fontsize=18)
    plt.xlabel("Características", fontsize=15)
    plt.ylabel("Muestras", fontsize=15)
    plt.tight_layout()
    if X.shape[1] >= 100:
        plt.axvline(x=100, color='yellow')
    plt.show()
    

    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='Spectral', center=0)
    plt.title('Similaridad entre columnas de la matriz de datos', fontsize=16)
    plt.xlim(1, 106)
    plt.ylim(106, 1)
    plt.show()

if __name__ == "__main__":
    main()
