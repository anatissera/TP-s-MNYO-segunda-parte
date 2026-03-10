## Repositorio de la segunda parte de la materia Métodos Numéricos y Optimización

### Trabajo Práctico 3: SVD y reducción de la dimensionalidad

En este trabajo se explora la aplicación de la Descomposición en Valores Singulares (SVD) y otras técnicas de reducción de dimensionalidad en el análisis de datos. La investigación se centra en la importancia de estas técnicas para simplificar conjuntos de datos complejos, mejorar su interpretabilidad y facilitar tareas como la visualización y la compresión de imágenes. Utilizando SVD y Análisis de Componentes Principales (PCA), se identifican las dimensiones más representativas de los datos y se evalúa su impacto en la precisión de modelos de predicción basados en cuadrados mínimos. Además, se investiga la eficiencia de la compresión de imágenes mediante SVD y representaciones aprendidas de otros conjuntos de datos. Los resultados muestran que la reducción de dimensionalidad no solo mejora la eficiencia computacional sino que también preserva la estructura esencial de los datos, permitiendo una mejor comprensión y manipulación de los mismos.

Desarrollo y análisis en [TP3_Informe.pdf](TP3/Informe.pdf) 

### Trabajo Práctico 4: Optimización

En este trabajo se investiga el uso del algoritmo iterativo de descenso por gradiente para resolver sistemas de ecuaciones lineales sobredeterminados. El estudio se enfoca en la minimización de la función de costo $F(x) = \|Ax - b\|_2^2$ y en la aplicación de regularización L2 para mejorar la estabilidad de las soluciones. Se generaron matrices $A$ y vectores $b$ aleatorios, y se aplicaron tanto el método de descenso por gradiente como la descomposición en valores singulares (SVD) para encontrar y comparar soluciones. Se analizó la influencia del tamaño de paso en la eficiencia y estabilidad del algoritmo de descenso por gradiente, destacando que el tamaño de paso $text{step size} = \frac{1}{\lambda_{\text{max}}}$ resulta ser el más efectivo. Además, se demuestra cómo la regularización L2 mejora la estabilidad y precisión de las soluciones obtenidas. De esta manera, el análisis proporciona una comprensión más profunda de las técnicas de optimización en sistemas lineales sobredeterminados.

Informe del trabajo en [TP4_Informe.pdf](TP4/Informe.pdf) 
