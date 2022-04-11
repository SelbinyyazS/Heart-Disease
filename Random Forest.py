import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


heart=pd.read_csv('heart.csv')
features= heart.iloc[:, :-1]
labels=heart.target

x_train, x_test, y_train, y_test=train_test_split(features, labels, test_size=0.2, random_state=42)
forest=RandomForestClassifier(n_estimators=400)

forest.fit(x_train, y_train)
print(forest.score(x_train, y_train))
print(forest.score(x_test, y_test))
