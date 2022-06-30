import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras import layers, models
import matplotlib as plt
import pandas as pd

train_ds = [1,2]
val_ds =[1,2]
test_ds = [1,2]

model = models.Sequential()
model.add(layers.LSTM(32, return_sequences=True, input_shape=train_ds[0].shape[1:])) # če hočemo več layerjev se več teh vrstic naredi
#Primer dodatnega layerja -> model.add(layers.LSTM(32, return_sequences=True))
model.add(layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x=train_ds[0], y=train_ds[1], validation_data=(val_ds[0], val_ds[1]), epochs=10) #epochs - pomeni kulkrat zažene model

x, y = test_ds
y_pred = model.predict(x)

y_pred.shape

#Plot results
fig, ax = plt.subplots()
i = 200
ax.plot(y[i:i+96*2,0], c='g')
ax.plot(y_pred[i:i+96*2,-1,0], c='r')


df_c = pd.DataFrame({'real': y[:,0], 'pred': y_pred[:, -1,0]})
df_c.corr()


model = models.Sequential()

model.add(layers.GRU(units=512,
              return_sequences=True,
              input_shape=(None, None,))) # dopolni input shape

model.add(layers.Dense(units=1))

model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')

model.summary()

model.fit(x=train_ds[0], y=train_ds[1], validation_data=(val_ds[0], val_ds[1]), epochs=10) #epochs - pomeni kulkrat zažene model

x, y = test_ds
y_pred = model.predict(x)

y_pred.shape

#Plot results
fig, ax = plt.subplots()
i = 200
ax.plot(y[i:i+96*2,0], c='g')
ax.plot(y_pred[i:i+96*2,-1,0], c='r')


df_c = pd.DataFrame({'real': y[:,0], 'pred': y_pred[:, -1,0]})
df_c.corr()