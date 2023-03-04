# In this file a nn created and trained with tensorflow will check if a given array is sorted or not.

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np 
import random

lines = 20
size = 10 

def constructLine():
	start = random.randint(0,9)
	line = list(range(start, start+size))
	unsort = random.randint(1,10)
	if unsort > 5:
		random.shuffle(line)
		line.append(0)
	else:
		line.append(1)
	return line

baseList = []
for line in range(0, lines):
	baseList.append(constructLine())
arr = np.array(baseList)

cols = []
for i in range(0, size):
	cols.append("Value" + str(i))
cols.append("sorted")


df = pd.DataFrame(arr, columns=cols)
X = pd.get_dummies(df.drop(["sorted"], axis=1))
Y = df["sorted"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

#Aufbau den Neuronalen Netzes
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

#Training des Models
model.fit(X_train, y_train, epochs=200, batch_size=32)

#Model verwenden und prüfen wie gut die Vorhersage ist
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(accuracy_score(y_test, y_hat))


print(X_test)
print(y_hat)