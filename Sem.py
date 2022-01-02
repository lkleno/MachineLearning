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
inputs_n = inputs.drop(['Material_Tip', 'Altern# kosovnice', 'Transportna skupina','Količina potreb','Osnovna količina', 'Količina','Obrat','Celotna količina naloga','Serialization Type','Nalog_H','Material1_H'],axis='columns')
#inputs_n = inputs.drop(['Material_Tip', 'Altern# kosovnice','Transportna skupina'],axis='columns')


from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(inputs_n,target,random_state=85,test_size=0.2)
test = x_valid['Kosovnica']
x_valid = x_valid.drop(['Kosovnica'],axis='columns')
x_train = x_train.drop(['Kosovnica'],axis='columns')
"""inputs_n = inputs_n.drop(['Kosovnica'],axis='columns')"""
from sklearn import tree
model = tree.DecisionTreeRegressor()

print("start learning")
#model.fit(inputs_n, target)
model.fit(x_train[:50000],y_train[:50000])
print("learned1")
print("learn2")
model.fit(x_train[50000:],y_train[50000:])
print("learned2")
print("ended learning")
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

from sklearn.model_selection import cross_val_score
decisionPred = model.predict(x_valid)
clf = tree.DecisionTreeRegressor()
print(np.mean(cross_val_score(clf,inputs_n,target,cv=5)))

from sklearn.decomposition import PCA

pca = PCA(0.95)
x_pca = pca.fit_transform(inputs_n)
print(x_pca.shape)
print(inputs_n.shape)

x_train,x_valid,y_train,y_valid = train_test_split(x_pca,target,random_state=85,test_size=0.2)

model = tree.DecisionTreeRegressor()

print("start learning")
#model.fit(inputs_n, target)
model.fit(x_train[:50000],y_train[:50000])
print("learned1")
print("learn2")
model.fit(x_train[50000:],y_train[50000:])
print("learned2")
print("ended learning")

#print(model.score(inputs_n, target))

#print(model.score(x_train, y_train))

print(model.score(x_valid, y_valid))

print("done")

#Evaluation

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

from sklearn.model_selection import cross_val_score
decisionPred = model.predict(x_valid)
clf = tree.DecisionTreeRegressor()
print(np.mean(cross_val_score(clf,inputs_n,target,cv=5)))
