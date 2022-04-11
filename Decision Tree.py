import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

heart=pd.read_csv('heart.csv')

#split the data into features and labels
#features are all columns except the last one (y=target)
#labels are only the last column (y=target)
features=heart.iloc[:, :-1]
labels=heart.iloc[:, -1]



x_train, x_test, y_train, y_test=train_test_split(features, labels, test_size=0.2, random_state=42)

#print(dct.score(x_train, y_train))
score=[]
for x in range(30):
  dct=DecisionTreeClassifier()
  dct.fit(x_train, y_train)
  score.append(dct.score(x_test, y_test))

plt.plot(range(0,30,1),score)
plt.show()
