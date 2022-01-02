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


print(x_valid.head)
print(y_valid.head)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
print("start learning")
model.fit(x_train[:10000],y_train[:10000])
print("learned1")
model.fit(x_train[10000:20000],y_train[10000:20000])
print("learned2")
model.fit(x_train[20000:30000],y_train[20000:30000])
print("learned3")
model.fit(x_train[30000:40000],y_train[30000:40000])
print("learned4")
model.fit(x_train[40000:50000],y_train[40000:50000])
print("learned5")
model.fit(x_train[50000:60000],y_train[50000:60000])
print("learned6")
model.fit(x_train[60000:70000],y_train[60000:70000])
print("learned7")
model.fit(x_train[70000:80000],y_train[70000:80000])
print("learned8")
model.fit(x_train[80000:90000],y_train[80000:90000])
print("learned9")
model.fit(x_train[90000:100000],y_train[90000:100000])

print("ended learning")
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