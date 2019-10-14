# SUV Prediction
import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("SUV_Predictions.csv")

dataset.head()

# Search in columns only the second and third columns (Age, EstimatedSalary)
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
