"""
De la misma forma en el Anexo L, podemos visualizar el algoritmo que nos ayudo a 
obtener las variables mas influyentes en el estado de resultados.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#import data
data = pd.read_csv("eeff.csv")
X = np.array(data.drop(['target'],1))
y = np.array(data['target'])
#etiquetado para los reportes graficos - > cambio de nombres 'mes' y 'cuenta'
data["mes"] = np.where(data["mes"]==1, "Enero", data["mes"])
data["mes"] = np.where(data["mes"]=="2", "Febrero", data["mes"])
data["mes"] = np.where(data["mes"]=="3", "Marzo", data["mes"])
data["mes"] = np.where(data["mes"]=="4", "Abril", data["mes"])
data["mes"] = np.where(data["mes"]=="5", "Mayo", data["mes"])
data["mes"] = np.where(data["mes"]=="6", "Junio", data["mes"])
data["mes"] = np.where(data["mes"]=="7", "Julio", data["mes"])
data["mes"] = np.where(data["mes"]=="8", "Agosto", data["mes"])
data["mes"] = np.where(data["mes"]=="9", "Septiembre", data["mes"])
data["mes"] = np.where(data["mes"]=="10", "Octubre", data["mes"])
data["mes"] = np.where(data["mes"]=="11", "Novienbre", data["mes"])
data["mes"] = np.where(data["mes"]=="12", "Diciembre", data["mes"])

data["cuenta"] = np.where(data["cuenta"]==1, "Venta Neta", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="2", "Costo de Venta", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="3", "Utilidad Bruta", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="4", "Gasto Administrativos", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="5", "Gasto Distrib. Ventas", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="6", "Gasto Operacional", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="7", "Gasto Personal", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="8", "Gasto Asistencia Social", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="9", "Gasto Educacional", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="10", "Otros Ingresos Operativos", data["cuenta"])
data["cuenta"] = np.where(data["cuenta"]=="11", "Utilidad Operativa", data["cuenta"])

#Efectividad del promedio (2014-2017) del estado de resultados con respecto a las partidas
pd.crosstab(data.cuenta, data.target).plot(kind="bar")
plt.title("Efectividad del promedio (2014-2017) del estado de resultados con respecto a las partidas")
plt.xlabel("Partidas del estado de resultados")
plt.ylabel("Efectividad del estado de resultados")
#Efectividad del promedio (2014-2017) del estado de resultados con respecto al año
pd.crosstab(data.anio, data.target).plot(kind="bar")
plt.title("Efectividad del promedio del estado de resultados con respecto al año")
plt.xlabel("Año")
plt.ylabel("Efectividad del estado de resultados")
#Efectividad del promedio (2014-2017) estado de resultados con respecto al mes
pd.crosstab(data.mes, data.target).plot(kind="bar")
plt.title("Efectividad del promedio (2014-2017) estado de resultados con respecto al mes")
plt.xlabel("Mes")
plt.ylabel("Efectividad del estado de resultados")
# converción de la variable objetivo a 0 y 1.
data["target"] = np.where(data["target"]==1, 1, data["target"])
data["target"] = np.where(data["target"]==2, 0, data["target"])
data["target"] = np.where(data["target"]==3, 0, data["target"])
#uniendo los features y los datos para obtener las varibles predictoras
categories = ["anio", "mes", "cuenta", "importe"]
for category in categories:
    cat_list = "cat" + "_" + category
    cat_dummies = pd.get_dummies(data[category], prefix=category)
    data_new = data.join(cat_dummies)
    data = data_new
data_vars = data.columns.values.tolist()
to_keep = [v for v in data_vars if v not in categories]
eeff_data = data[to_keep]
eeff_data_vars = eeff_data.columns.values.tolist()
Y = ['target'] #varible objetivo
X = [v for v in eeff_data_vars if v not in Y] #features
n = 12 # catidad de variables que se quiere obtener
lr = LogisticRegression(solver='liblinear')
rfe = RFE(lr, n)
rfe = rfe.fit(eeff_data[X], eeff_data[Y].values.ravel())
z=zip(eeff_data_vars,rfe.support_, rfe.ranking_)
print(list(z)) #seleccionamos varilaes con notacion (..., True, 1)
#una vez seleccionado las varibles lo almacenamos en la varible 'clos'
cols = ["anavertical","anio_2016","anio_2015",
	   "cuenta_Gasto Administrativos",
	    "cuenta_Gasto Distrib. Ventas",
	    "cuenta_Gasto Operacional",
	    "cuenta_Gasto Personal",
        "cuenta_Otros Ingresos Operativos",
	   "cuenta_Utilidad Operativa",
	   "mes_Abril",
	   "mes_Agosto",
	   "mes_Octubre",
	   "mes_Septiembre"]
X = eeff_data[cols]
Y = eeff_data["target"] # variable objetivo
#Validación de resultados entre los dos modelos
statsmodels = sm.Logit(Y, X)
result = statsmodels.fit()
# Resultado STATSMODELS
print(result.summary2())
logit_model = linear_model.LogisticRegression(solver='liblinear')
logit_model.fit(X,Y)
# resultado SKLEARN
print(pd.DataFrame(list(zip(X.columns, np.transpose(logit_model.coef_)))))






