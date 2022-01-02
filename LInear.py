import pandas as pd

df = pd.read_csv("KopijaKopijaPodatki_sem_v6_NH.csv", delimiter=';')

inputs = df.drop(['Odvzeta količina'],axis = 'columns')
target = df['Odvzeta količina']

from sklearn.preprocessing import LabelEncoder

le_matTip = LabelEncoder()
le_altKos = LabelEncoder()
le_transSkup = LabelEncoder()

inputs['Material_Tip_n'] = le_matTip.fit_transform(inputs['Material_Tip'])
#inputs['Altern# kosovnice_n'] = le_altKos.fit_transform(inputs['Altern# kosovnice'])
#inputs['Transportna skupina_n'] = le_transSkup.fit_transform(inputs['Transportna skupina'])

inputs_n = inputs.drop(['Material_Tip', 'Altern# kosovnice', 'Transportna skupina','Količina potreb','Osnovna količina','Kosovnica', 'Količina','Obrat','Celotna količina naloga','Serialization Type','Nalog_H','Material1_H'],axis='columns')

from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(inputs_n,target,random_state=85,test_size=0.2)


from sklearn import linear_model

model = linear_model.LinearRegression()
print("start learning")
model.fit(x_train[2:],y_train[2:])
print("learned")
print(model.score(x_valid,y_valid))
print(model.score(x_train,y_train))

from sklearn.model_selection import cross_val_score
import numpy as np

clf = linear_model.LinearRegression()
print(np.mean(cross_val_score(clf,inputs_n,target,cv=10)))

import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score
gt = y_valid
pred = model.predict(x_valid)

arr = []
for x in pred:
    arr.append(round(x))
arr2 = []
for y in gt:
    arr2.append(y)

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
print('r2 score:', r2_score(gt,pred))
mape = np.mean(np.abs((gt-pred)/np.abs(gt)))
print(mape)
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
