import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("16-diabetes.csv")
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df['Insulin'].value_counts())#374 tane 0 degeri insulinden gelmis

columns_to_check=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in columns_to_check:
    zero_count=(df[col]==0).sum()
    print(zero_count)

X=df.drop('Outcome',axis=1)
y=df['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

columns_to_fill = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

median = {}

for col in columns_to_fill:
    median_value = X_train.loc[X_train[col] != 0, col].median()
    median[col] = median_value

for col in columns_to_fill:
    X_test[col] = X_test[col].replace(0, median[col])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

print("RANDOM FOREST confusion_matrix VALUES :",confusion_matrix(y_test,y_pred))
print("RANDOM FOREST classification_report VALUES :",classification_report(y_test,y_pred))
print("RANDOM FOREST accuracy_score VALUES :",accuracy_score(y_test,y_pred))


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


print("K - NEIGHBORS confusion_matrix VALUES :",confusion_matrix(y_test,y_pred))
print("K - NEIGHBORS classification_report VALUES :",classification_report(y_test,y_pred))
print("K - NEIGHBORS accuracy_score VALUES :",accuracy_score(y_test,y_pred))

classifier=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


print("DESCISION TREE confusion_matrix VALUES :",confusion_matrix(y_test,y_pred))
print("DESCISION TREES classification_report VALUES :",classification_report(y_test,y_pred))
print("DESCISION TREE accuracy_score VALUES :",accuracy_score(y_test,y_pred))

#hyperparametre

tree_param={
    "criterion":["gini","entropy","log_loss"],
    "splitter":["best","random"],
    "max_depth":[1,2,3,4,5,15,None],
    "max_features":["sqrt", "log2",None]
}
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=tree_param,cv=5,scoring="accuracy")
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)

print("HYPARAM DESCISION TREE confusion_matrix VALUES :",confusion_matrix(y_test,y_pred))
print("HYPARAM DESCISION TREES classification_report VALUES :",classification_report(y_test,y_pred))
print("HYPARAM DESCISION TREE accuracy_score VALUES :",accuracy_score(y_test,y_pred))

rf_param={
    "n_estimators": [100, 200, 300, 1000],
    "max_depth": [5, 8, 10, 15, None],
    "max_features": ["sqrt", "log2", 5, 6, 7, 8],
    "min_samples_split": [2, 8, 15, 20]
}

from sklearn.model_selection import RandomizedSearchCV
rfsc=RandomizedSearchCV(estimator=rf,param_distributions=rf_param,cv=3,n_jobs=-1)
rfsc.fit(X_train,y_train)
y_pred=rfsc.predict(X_test)

print("HYPARAM RANDOM FOREST confusion_matrix VALUES :",confusion_matrix(y_test,y_pred))
print("HYPARAM RANDOM FOREST classification_report VALUES :",classification_report(y_test,y_pred))
print("HYPARAM RANDOM FOREST accuracy_score VALUES :",accuracy_score(y_test,y_pred))


knn_param={ 
    "n_neighbors":[1,2,3,4,5],
    "weights":["uniform", "distance",None],
    "algorithm":["auto", "ball_tree", "kd_tree", "brute"]
}
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn=GridSearchCV(KNeighborsClassifier(),param_grid=knn_param,cv=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


print("HYPARAM K - NEIGHBORS confusion_matrix VALUES :",confusion_matrix(y_test,y_pred))
print("HYPARAM K - NEIGHBORS classification_report VALUES :",classification_report(y_test,y_pred))
print("HYPARAM K - NEIGHBORS accuracy_score VALUES :",accuracy_score(y_test,y_pred))