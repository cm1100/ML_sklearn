import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("Social_Network_Ads.csv")

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

clf = SVC(kernel="rbf",)
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)

print(acc)


accuracies = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10)
print(accuracies.std())



