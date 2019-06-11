"""
En el Anexo K, podemos visualizar el algoritmo que nos ayudo a proyectar el una partida del estado
de resultados, llamado utilidad operativa, en donde nosotros entregamos nuevos valores al algoritmo,
una vez procesado el algoritmo nos da un resultado de una efectividad buena o mala.
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#import data
data = pd.read_csv("eeff.csv")
X = np.array(data.drop(['target'],1))
y = np.array(data['target'])

##alimetamos con el 80% de la data para el entrenamiento y el 20% para validar
model = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=7)
model.fit(X_train, y_train)
print("score del modelo cross validation:",model.score(X, y))

#K-Fold Cross Validation
name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=7)
cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
predictions = model.predict(X_test)

##validaciones
print(accuracy_score(y_test, predictions))#cross validation
print(confusion_matrix(y_test, predictions))#matriz de confucion
print(classification_report(y_test, predictions))# reporte de clasificación

#predicción de nuevos valores
X_new = pd.DataFrame({
	'anio': [2019], 
	'mes': [2], 
	'cuenta': [11], 
	'importe': [-232232676.99], 
	'anavertical': [0.1322]})
model.predict(X_new)
print("Prediccion::",model.predict(X_new))
print("porcentaje de aceptacion",accuracy_score(y_test, predictions)*100,"%")

