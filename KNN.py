from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

heart=pd.read_csv('heart.csv')
features=heart.iloc[:, :-1]
labels=heart.iloc[:, -1]

x_train, x_test, y_train, y_test=train_test_split(features, labels, test_size=0.2, random_state=42)

ct=ColumnTransformer([('scale',StandardScaler(),features.columns)])
x_train=ct.fit_transform(x_train)
x_test=ct.transform(x_test)




score_list=[]
for i in range(1, 10):
  model=KNeighborsClassifier(n_neighbors=i)
  model.fit(x_train, y_train)
  score_list.append(model.score(x_test, y_test))

plt.plot(range(1,10,1), score_list)
plt.show()
print("max:", max(score_list))




model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
model.score(x_test, y_test)

 
