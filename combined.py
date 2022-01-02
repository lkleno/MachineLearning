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


from sklearn.model_selection import cross_val_score
import numpy as np
#Lin model
from sklearn import linear_model
print("lin model")
model = linear_model.LinearRegression()
print("start learning")
model.fit(x_train[2:],y_train[2:])
print("learned")
print(model.score(x_valid,y_valid))
linearPred = model.predict(x_valid)
clf = linear_model.LinearRegression()
cvLin=(np.mean(cross_val_score(clf,inputs_n,target,cv=5)))

#decision tree
from sklearn import tree
model = tree.DecisionTreeRegressor()
print("decision tree")
print("start learning")
#model.fit(inputs_n, target)
model.fit(x_train[:50000],y_train[:50000])
print("learned1")
print("learn2")
model.fit(x_train[50000:],y_train[50000:])
print("learned2")
print("ended learning")
decisionPred = model.predict(x_valid)
clf = tree.DecisionTreeRegressor()
cvDecTree = (np.mean(cross_val_score(clf,inputs_n,target,cv=5)))

#k neigh
from sklearn.neighbors import KNeighborsRegressor
print("k neigh")
model = KNeighborsRegressor(n_neighbors=1)
print("start learning")
model.fit(x_train[:50000],y_train[:50000])
print("learned1")
print("learn2")
model.fit(x_train[50000:],y_train[50000:])
print("learned2")
print("ended learning")
print(model.score(x_valid,y_valid))
kPred = model.predict(x_valid)

clf = KNeighborsRegressor(n_neighbors=1)
cvKNeigh = (np.mean(cross_val_score(clf,inputs_n,target,cv=5)))

#random forest
from sklearn.ensemble import RandomForestRegressor
print("random forest")
model = RandomForestRegressor(n_estimators=100, warm_start=False)
print("started learning")
model.fit(x_train[:25000],y_train[:25000])
print("learned")
#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))
print("learn2")
model.n_estimators += 100

model.fit(x_train[25000:50000],y_train[25000:50000])
print("learned2")
#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))

model.n_estimators += 100
print("learn3")
model.fit(x_train[50000:75000],y_train[50000:75000])
print("learned3")

#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))

model.n_estimators += 100
print("learn4")
model.fit(x_train[75000:],y_train[75000:])
print("learned4")

RFPred = model.predict(x_valid)
clf = RandomForestRegressor(n_estimators=100, warm_start=False)
cvRF = (np.mean(cross_val_score(clf,inputs_n,target,cv=5)))

#SVM
from sklearn.svm import LinearSVR
model = LinearSVR(dual=False,loss='squared_epsilon_insensitive')
print("svm")
#print("started learning")
#model.fit(x_train,y_train)

print("started learning")
model.fit(x_train[:25000],y_train[:25000])
print("learned")
#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))
print("learn2")


model.fit(x_train[25000:50000],y_train[25000:50000])
print("learned2")
#print(model.score(x_train,y_train))
#print(model.score(x_valid,y_valid))


print("learn3")
model.fit(x_train[50000:75000],y_train[50000:75000])
print("learned3")


print("learn4")
model.fit(x_train[75000:100000],y_train[75000:100000])
print("learned4")

print("learned")
SVMPred = model.predict(x_valid)
clf = LinearSVR(dual=False,loss='squared_epsilon_insensitive')
cvSVM = (np.mean(cross_val_score(clf,inputs_n,target,cv=5)))

print("Line reg", cvLin)
print("SVM", cvSVM)
print("k neigh", cvKNeigh)
print("Decision tree", cvDecTree)
print("Random forest", cvRF)
import matplotlib.pyplot as plt

plt.scatter(test[:100],linearPred[:100], marker='o',color='b',label='Linear regression')
plt.scatter(test[:100],decisionPred[:100], marker='o',color='#F48D3C',label='Decision tree')
plt.scatter(test[:100],kPred[:100], marker='o',color='#3CF4EC',label='K-nearest neighbor')
plt.scatter(test[:100],RFPred[:100], marker='o',color='#F43CF4',label='Random forest')
plt.scatter(test[:100],SVMPred[:100], marker='o',color='#F4E03C',label='Support vector machine')
plt.scatter(test[:100],y_valid[:100],color='r',marker='o',label='Actual value')
plt.xlabel("Unique ID")
plt.ylabel("Order quantity")
plt.legend()
plt.show()