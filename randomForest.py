import pandas as pd

df = pd.read_csv("KopijaKopijaPodatki_sem_v5_NH.csv", delimiter=';')

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


fn = list()
for col in inputs_n.columns:
    fn.append(col)

print(fn)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50, warm_start=False)
print("started learning")
model.fit(x_train[:25000],y_train[:25000])
print("learned")
#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))
print("learn2")
model.n_estimators += 50

model.fit(x_train[25000:50000],y_train[25000:50000])
print("learned2")
#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))

model.n_estimators += 50
print("learn3")
model.fit(x_train[50000:75000],y_train[50000:75000])
print("learned3")

#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))

model.n_estimators += 50
print("learn4")
model.fit(x_train[75000:],y_train[75000:])
print("learned4")

#print(model.score(x_train,y_train))
print(model.score(x_valid,y_valid))

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

import matplotlib.pyplot as plt
'''RFPred = model.predict(x_valid)
plt.scatter(test[:100],RFPred[:100], marker='o',color='#F43CF4',label='Random forest')
plt.scatter(test[:100],y_valid[:100],color='r',marker='o',label='Actual value')
plt.xlabel("Unique ID")
plt.ylabel("Order quantity")
plt.legend()
plt.show()'''


import time
import numpy as np
print()
print()
print()
print()
print()
print()
start_time = time.time()
importances = model.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
import pandas as pd

#forest_importances = pd.Series(importances, index=fn)

fig, ax = plt.subplots()
#forest_importances.plot.bar(yerr=std, ax=ax)
plt.scatter(importances,fn, marker='o',color='#F43CF4',label='Random forest')
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()