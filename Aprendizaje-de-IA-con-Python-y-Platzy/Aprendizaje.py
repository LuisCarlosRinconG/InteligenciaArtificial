# Un feature es el proceso previo a la creación de un modelo que consistte en hacer análisis, limpieza y estructuración de datos
# la libreria pandas ayuda a cargar y analizar datos
# los histogramas ayudan a ver la desttribución en feature
# Las graficas de dispersión permitten ver la relación enttre dos feattures


'''
Tipos de modelos de machine learning

+ Aprendizaje supervisado: predicón un objetivo o target(Con un objetivo)
El modelo obtiene features de entrada y salida, Hay un target/objetivo a predecir

- Regresión: Target output(Objetivo de salida) es numerico
-Clasificasión: Target output es una etiqueta

+ Aprendizaje no supervisado: Predicen patrones o estructuras en los datos(Sin un objetivo)
objetivo desconocido, queremos encontrar estructura y grupos denttro de los datos

- Clustering: Queremos enconttrar grupos de datos
- Dimensionality reduction: Queremos encontrara que features de entrada en los datos son de ayuda

'''


'''
# Paquetes numéricos
import numpy as np

# DataFrames/Procesamiento
import pandas as pd

# Gráficas
import matplotlib.pyplot as plt

'''

'''

¿Como utilizar modelos de machine learning?

1)ingredientes de los algoritmos de machine learning

-Proceso de decisión: Como los modelos hacen prediccion, o retornan una respuesta, usando los parámetros
-Función de error/costte: Cómo evaluar so los paramteros en el modelo generan buenas predicciones.
-Regla de actualización: Cómo mejorar los paramettros para hacer mejores predicciones(usano optimización numérica),

-Normalizar tus datos: Generar estabilidad numerica(Facilidad del modelo para utilizar los datos)

-Prepara tus datos para modelos:
--Training:(60-80%) Datos de los que el modelo aprende patrtones
--Validation:(0-20%) Datos que usas para verificar que el modelo aprende
--Testing:(0-20%) Datos que se apartan para revisar si el modelo fue exitoso al predecir


2)Algoritmos de aprendizaje supervisado

-regresión lineal

--Lineal - positiva: Mientras x crece y crece la linea en medio del plano x,y tambien lo hace
--Lineal - negattiva: Mientras X crece, y Y disminuye la linea en medio del plano x,y decrece
--No lineal

-Regresión lineal

---Proceso de decisión: Una función para predecir el target de salida en features de entrada

---Función de error/coste: (Valor de que tan bien se predice el target) Es la diferencia entre el valor de target y el valor predecido de nuestra predicción

--- Regla de actualización:(Encuenttra los valores que mejoren las predicciones) Cuando se quiere minimizar la distancia de la prediccion sobre cada puntto de datos en entrenamiento

--- Error cuadratico: (se busca un valor cercano a uno)Es la correlaciónn enttre los valores de entrada y salida y va de la mano
con el rendimiento que sirve para identificar la fortaleza de la relación entre features de entrada y de salida

-Reggresión logística: Técnica de análisis de datos que utiliza las mattematicas para encontrar las relaciones enttre dos factores de datos

-Diferencia entre regresion logistica y lineal: La regresión lineal sigue una distribución normal o gaussiana de la variable dependiente. Una distribución normal se representa 
mediante una línea continua en un gráfico. Una regresión logística sigue una distribución binomial. La distribución binomial se suele representar como un gráfico de barras
ultimo fragmento sacado de: Regresión lineal frente a regresión logística: diferencia entre las técnicas de machine learning - AWS. (s. f.). Amazon Web Services, 
Inc. https://aws.amazon.com/es/compare/the-difference-between-linear-regression-and-logistic-regression/#:~:text=La%20regresi%C3%B3n%20lineal%20sigue%20una,como%20un%20gr%C3%A1fico%20de%20barras.

-Random forest: Un bosque aleattorio es un conjunto o un grupo de decisiones que votan por la respesta correcta
-- Arboles de desición: Sirven para etiquetar datos y tomar decisiones mediante el si o no
--Elementos de random forest
---Numero de arboles: Mas arboles menor la variación, pero más c{omputo}
---Max features: El número de feattures usados para partir(Split)
---Max depth: El numero de niveles del arbol de decisión
-- Paramettros de random forest
--- N =  min samples split: Numero de data points que un nodo debe tener antes de dividirse
--- n = Min samples leaf: El minimo número de muestras requeridas para estar en una hoja (leafa)

-- Se pueden usar métricas de clasificación refresión para evaluar qué tan bueno es el modelo

3)Algoritmos de aprendizaje no supervisado

-K-means: El objetivo de este modelo es encontrar que puntos de los datos se asignan a que grupos
-- Centroide: Son pocisiones en el espacio que representan cada una de los features de entrada y a menudo se colocan al azar aunque hay formas de no hacerlo
-- Rendimiento:
--- Inertia: Qué tan cerca están los punttos de datos al centroid. Este número se necesita pequeño
--- Silhouettte score: Qué ttan lejanos son los clusters, de[-1,1]. Estet numero debe ser cercano a 1
-- En este modelo se buca un grafico de codo y especialmente lo que se busca en el es la curva en la trama(velocidad de cambio más lenta para números de clusters)
-- Proceso de decision de k- means: 
---Calcular la distancia de cada punto de dattos a cada centtroide.
---Asignar cada punto de dattos al centroide más cercano
---Calcular los nuevos centroides promediando los puntos de datos asignados al clustter
'''

'''
Deep learning
Redes neuronales: Una red neuronal es un modelo que usa neuronas y conexiones entre ellos para hacer predicciones y son usadas usualamente para parendizaje supervisado y usualmente se habla de ellas en un conttexto de regresión o de clasificación

Las redes neuronales tienen:
- capa de entrada: Que toma los datos a menudo pre-procesados, y los alimenta en unna capa oculta 
- capa oculta: pueden haber una o varias capas ocultaas y representan las operaciones y funciones complejas qque permiten modelar preguntas
- Capa de salida: nos proporciona la predicción o respuesta 
-Cada uno de los modulos anteriores se consideran nodos o neuronas y sonn operaacionnes o funciones que toman los datos de cada capa individual hasta la caapa de salida.
-Entrte los nodos existe algo llamado conexión o borde e indican la ffuerza de la relación enttre los nodos

- unidad oculta: los pesos gobierna la fuerza de una conexión entre nodos. Pessos altos == conexión fuerte  
-- La unidad escondida(nodo) aceptta una combinación linnela de nodos adjutntos previamenttte a él
-- Cada unidad oculta recibe unna combinación lineal en todas las entradas 
-- Lla unnidad oculta ejecuta una función denn la combinación lineal. Estta función es una función de activación
--- Algunas funciones de activación tienen un rango limittado mientras que otras se xtienden indefinidamente

- ¿Que es deep learning?: Es introducir profundidad a las capas oculatas de la red nneuronal
-- Profundidad: Agregar masa capas
-- Ancho: Agregar mas unidades oculttas 

Tipos de redes neuronales:
-Existen varios tipos pero tttres de las mas comunes son: 
--redes neuronales profundas de avance o redes neuronales profundas: Es usada en muchos problemas complejos
--redes neuronales convolucionales: usadas en imaggenes y genómicos
-- redes neuronales recurrentes: Represeta secuencias y es usada en lenguaje

-Una vez que se preparan los datos y se inicializan los pesos se ttiene un proceso 
- Receta de una red neuronal:(Entrenar involucra)
--Regla de decisión: es un calculo de avance  
-- Funcion de perdida de error: mide lo bien que la red ffue capaz de predecir un vallor dada la verdad de base o vallor real en la deude de entrenamiento 
-- Propagación haacia atras: Es la regla de aactualización reaal de las redes neuronales y ayudaará a ajusttaar llos muchos pesos de la red neuronal para poder tener la mejor predicción posible

-Las mejores formasa de mejorar una redd neuronal:
-- Pregutaar especifficamente si la perdida esta camiando. la perdida en una forma de decir lo bien de las predicciones y coincidenn con lo que se espera del valor real y a medida que la perdida se haace mas pequeña el modelo estta entrenado mejor 
-- se debe tenner cuidado con sobreadaptar una red que es cuando esta aprendio las reglas de forma muy especifica para los datos de entrenamiento y ahí es donde es muy importante un conjunto de ddatos de validación 

- Procesos
--Avance hacia adelante: valuar las funciones de cada nodo

-- Error cuadratico medio en la funcion de perdidad: para regresión. Diferencia en valor verdadero versus predecido
-- Para la clisificación se llama perdida de entropia cruzada binaría  y se calscula qque ttan confiable el modelo predice la probabilidad de una clase para 2 clases
-- Las anteriroes se pueden generalizar en clasificación de ttipo multi o categorica a ttraves de las perdidas para mas de dos ettiquetas

--propagación hacia atras: es la "regla de actualización" usada para ajustar los pesos en redes neuronales
--Para la propagación hacia atras es utilizada la llamada derivada parcial para ver cuanto cambiará la salida con un pequeño cambio en la actualización de el objetivo que tiene que ver con cuánto cambiara la salida con un pequeño cambio en la actualización de un peso partticular 


Como mejorar las predicciones de redes neuronales:
-La tasa de aprendizaje y drpout son importantes para el entrenamiento
-revisar la pérdida y el rendimiento en el set de validación








'''
'''
Credittos a: https://platzi.com/
'''