import pandas as pd

df = pd.read_csv("KopijaKopijaPodatki_sem_v4_NH.csv", delimiter=';')

inputs = df.drop(['Dejanska količina naloga'],axis = 'columns')
target = df['Dejanska količina naloga']

from sklearn.preprocessing import LabelEncoder

le_matTip = LabelEncoder()
le_altKos = LabelEncoder()
le_transSkup = LabelEncoder()

inputs['Material_Tip_n'] = le_matTip.fit_transform(inputs['Material_Tip'])
inputs['Altern# kosovnice_n'] = le_altKos.fit_transform(inputs['Altern# kosovnice'])
#inputs['Transportna skupina_n'] = le_transSkup.fit_transform(inputs['Transportna skupina'])

inputs_n = inputs.drop(['Material_Tip', 'Altern# kosovnice', 'Transportna skupina'],axis='columns')


from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(inputs_n,target,random_state=85,test_size=0.2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

print("start learning")
model.fit(x_train[:50000],y_train[:50000])
print("learned1")
print("learn2")
model.fit(x_train[50000:],y_train[50000:])
print("learned2")
print("ended learning")
print(model.score(x_valid,y_valid))

from sklearn.decomposition import PCA

pca = PCA(0.95)
x_pca = pca.fit_transform(inputs_n)
print(x_pca.shape)
print(inputs_n.shape)


model = LogisticRegression()

print("start learning 2")
model.fit(x_train[:50000],y_train[:50000])
print("learned1")
print("learn2")
model.fit(x_train[50000:],y_train[50000:])
print("learned2")
print("ended learning")
print(model.score(x_valid,y_valid))