import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# Separar conjunto de entrenamiento y de validacion
from sklearn.model_selection import train_test_split
# se empiezan a construir los clasificadores con arboles de decisión 
from sklearn.tree import DecisionTreeClassifier
# Utilizaremos random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from matplotlib import colors
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest

 
# *****************************************************************************************************************************************

# Random forest


# Se carga el conjunto de datos spam.data 
## sep== separacion por comas
## Header== no tiene
df=pd.read_csv('csv_Analisis_data_tesis_2.1.csv', sep=',', header=None)

# separa los datos en "XRF" y en "YRF"
# "XRF" representa todas las columnas menos la ultima la cual es la de respuesta
# "YRF" representa la columna de respuesta

XRF = df.iloc[:,:-1]
YRF = df.iloc[:,-1]

# Normalizamos los datos con logaritmos neperianos
XRF_normalizada=np.log1p(df.iloc[:,:-1])

# configuracion dl modelo con datos normalizados mediante a logaritmos neperianos
XRF_train, XRF_test, YRF_train, YRF_test = train_test_split(XRF_normalizada,YRF, test_size=0.2, random_state=1)

#Random forest

rf= RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=1000,min_samples_split=3,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='sqrt',max_leaf_nodes=10,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=-1,random_state=1,verbose=0,warm_start=False,ccp_alpha=0.0,max_samples=10)

rf.fit(XRF_train,YRF_train)

YRF_prdss = rf.predict(XRF_test)

#Indice de ocurrencia
ocurrenciaRF=accuracy_score(YRF_test, YRF_prdss)
print("El porcentaje de efectividad de random forest es: ", ocurrenciaRF)


# se crea un dataframe para comparar con lso valores reales
print("* Datos de Randon Forest")
rtaRF = pd.DataFrame({'real': YRF_test,'predicciones': YRF_prdss})


lista=[YRF_test]

print(rtaRF)
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# #Este array cpnprueba que los datos se almacenan como un solo valor 
print(lista[0])
# Por lo tanto la incongnita es como dividir los datos uno por uno
# +++++++++++++++++++++++++++++++++++++++++++++++++++




# *******************************************************************************************************************************************************************
# Support Vector Machines
datos=pd.read_csv('csv_Analisis_data_tesis_2.1.csv', sep=',', header=None)

# separa los datos en "XMVS" y en "YMVS"
# "XMVS" representa todas las columnas menos la ultima la cual es la de respuesta
# "YMVS" representa la columna de respuesta

YMVS = datos.iloc[:,-1]
XMVS = datos.iloc[:,:-1]



# División de los datos en un 80% para entrenamiento y un 20% para prueba
XMVS_train,XMVS_test, YMVS_train, YMVS_test = train_test_split(XMVS, YMVS, test_size=0.2, random_state=42)

clf = SVC(kernel = 'linear').fit(XMVS_train, YMVS_train)
ocurrenciaMVS=clf.score(XMVS_test, YMVS_test)
print("El porcentaje de efectividad del modelo Support Vector Machines es: ", ocurrenciaMVS)

YMVS_pred = clf.predict(XMVS_test)

# Calculo de la matriz de correlación
correlation_matrix = datos.corr()

print("* Datos de Support Vector Machines")
RTAMVS = pd.DataFrame({'Real': YMVS_test,'Predicho': YMVS_pred})

print(RTAMVS)


# ***************************************************************************************************************************************



# Modelo Naive Bayes


# separa los datos en "XNV" y en "Y"
# "XNV" representa todas las columnas menos la ultima la cual es la de respuesta
# "Y" representa la columna de respuesta


df_data = pd.read_csv('csv_Analisis_data_tesis_2.1.csv', sep=',', header=None)

# Haremos Feature Selection para mejorar los resultados del algoritmo. Utilizando la clase SelectKBest para seleccionar las 5 mejores caracteristicas. Son las variables que mas aportan al momento de realizar la clasificación.

XNV = df_data.iloc[:,:-1]
y = df_data.iloc[:,-1]

best = SelectKBest(k = 5)
XNV_new = best.fit_transform(XNV,y)
XNV_new.shape
selected = best.get_support(indices = True)

used_features = XNV.columns[selected]

# División de los datos en conjuntos de prueba y entrenamiento, 80% para entrenamiento y 20% para prueba

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 6)
XNV_train, XNV_test, y_train, y_test = train_test_split(XNV,y, test_size = 0.2, random_state = 6)


# Normalizacion de la data

from sklearn.naive_bayes import GaussianNB

bayes_naive = GaussianNB()
bayes_clasif = bayes_naive.fit(XNV_train[used_features].values, y_train)
y_pred = bayes_naive.predict(XNV_test[used_features])
#y_pred = bayes_clasif.predict(X_test)

#Indice de ocurrencia
ocurrenciaNV=accuracy_score(y_test, y_pred)

print("El porcentaje de efectividad de Naive Bayes es: ", ocurrenciaNV)

RTANV = pd.DataFrame({'Real': y_test,'Predicho': y_pred})

print(RTANV)
