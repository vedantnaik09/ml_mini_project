

import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/content/Dataset_spine.csv')

data.head()

data.info()

data.shape

del data["Unnamed: 13"]

data.describe()

data.rename(columns = {
    "Col1" : "pelvic_incidence",
    "Col2" : "pelvic_tilt",
    "Col3" : "lumbar_lordosis_angle",
    "Col4" : "sacral_slope",
    "Col5" : "pelvic_radius",
    "Col6" : "degree_spondylolisthesis",
    "Col7" : "pelvic_slope",
    "Col8" : "direct_tilt",
    "Col9" : "thoracic_slope",
    "Col10" :"cervical_tilt",
    "Col11" : "sacrum_angle",
    "Col12" : "scoliosis_slope",}, inplace=True)

data.head()

data.shape

data["Class_att"].value_counts().sort_index().plot.barh()

import matplotlib.pyplot as plt

for column in data.columns:
  plt.figure()
  sns.histplot(data[column])
  plt.title(f"Distribution of {column}")
  plt.show()

sns.pairplot(data)
plt.show()

corr_matrix = data.iloc[:, :-1].corr()
print(corr_matrix)

sns.heatmap(data[data.iloc[:, :-1].columns[0:13]].corr(),annot=True,cmap='viridis',square=True, vmax=1.0, vmin=-1.0, linewidths=0.2)

data=data.drop_duplicates()
data.head()

data=data.drop(data[data['degree_spondylolisthesis']>400].index,axis=0)

X = data.drop(data.columns.tolist()[-1], axis=1)
X.head()

le = LabelEncoder()
data['Class_att'] = le.fit_transform(data['Class_att'])
data.sample(5)

Y = data['Class_att']
Y.head()

model, acc = [], []

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
_ = decision_tree.fit(X_train, Y_train)
_ = decision_tree.predict(X_test)
accuracy = decision_tree.score(X_test, Y_test)
print(accuracy)
model.append("Decision Tree")
acc.append(accuracy)

from sklearn.linear_model import LogisticRegression
logres = LogisticRegression()
logres.fit(X_train, Y_train)
accuracy = logres.score(X_test, Y_test)
print(accuracy)
model.append("Logistic Regression")
acc.append(accuracy)

from sklearn.ensemble import RandomForestClassifier
randforest = RandomForestClassifier()
randforest.fit(X_train, Y_train)
accuracy = randforest.score(X_test, Y_test)
print(accuracy)
model.append("Random Forest")
acc.append(accuracy)

_=sns.barplot(x = model , y = acc ,palette ='Spectral')