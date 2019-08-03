import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm

dataset_train = pd.read_csv('train_tit.csv')

dataset_test = pd.read_csv('test_tit.csv')



data = {'S':  1,'C':0}

dataset_train['Embarked'] = dataset_train['Embarked'].map(data)



dataset_train['Name'] = pd.to_numeric(dataset_train['Name'], errors='coerce')

dataset_train['Sex'] = pd.to_numeric(dataset_train['Sex'], errors='coerce')

dataset_train['Ticket'] = pd.to_numeric(dataset_train['Ticket'], errors='coerce')

dataset_train['Cabin'] = pd.to_numeric(dataset_train['Cabin'], errors='coerce')



x = dataset_train.iloc[:,0:11]

x = x.fillna(0)

y = dataset_train[dataset_train.columns[-1]]

y = y.fillna(0)




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=50)


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=50)

random_forest.fit(X_train, y_train.ravel())

predict = random_forest.predict(X_test)

from sklearn import metrics
print("Accuracy  = {0:3f}".format(metrics.accuracy_score(y_test, predict)))




















