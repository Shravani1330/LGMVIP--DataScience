# LGMVIP--DataScience(JAN 2023)

NAME- SHRAVANI UPADHYAY

TASK1- IRIS FLOWERS CLASSIFICATION

Dataset Link-http://archive.ics.uci.edu/ml/datasets/Iris

IMPORTING LIBRARIES

IMPORTING LIBRARIES

#Importing all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

LOADING THE DATASET
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
               names=["sepal_length_in_cm","sepal_width_in_cm","petal_length_in_cm","petal_width_in_cm","class"])
               df.head()
               
DATA ANALYSIS
df.info()
df.describe()

DATA VISUALIZATION
fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot (x='sepal_length_in_cm',y='petal_length_in_cm',data=df,hue='class',ax=ax1)
sns.scatterplot (x='sepal_width_in_cm',y='petal_width_in_cm',data=df,hue='class',ax=ax2)


sns.pairplot(df,hue="class")
sns.heatmap(df.corr(),annot=True)

BUILDING, TRAINING AND TESTING THE MODEL
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['class']=le.fit_transform=(df['class'])
df.head()

PREPARING THE DATA FOR TRAINING
X=df[['sepal_length_in_cm','sepal_width_in_cm','petal_length_in_cm','petal_width_in_cm']]
Y=df['class']

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test,  Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print('Accuracy of the logistic regression on the test set:{:f}'.format(model.score(X_test,Y_test)))

from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))

from sklearn.metrics import accuracy_score
train_score=str(model.score(X_train,Y_train)*100)
test_score=str(model.score(X_test,Y_test)*100)
accu_score=str(accuracy_score(Y_test,y_pred)*100)
print(f'Train score:{train_score[:6]}%\n Test score:{test_score[:6]}%\n Accuracy score:{accu_score[:6]}%')
