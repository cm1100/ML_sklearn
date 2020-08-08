import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score


df = pd.read_csv("Social_Network_Ads.csv")
#print(df.head())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
pred = clf.predict(X_test)

acc1 = accuracy_score(pred,y_test)
print(acc1)

mat1 = confusion_matrix(y_test,pred)
print(mat1)