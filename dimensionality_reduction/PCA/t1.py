import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("Wine.csv")

#print(list(df))
#print(df.head())

X = df.iloc[:,:-1].values
y= df.iloc[:,-1].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X_train1,X_test1= X_train,X_test

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_train1=sc.fit_transform(X_train1)
X_test1=sc.fit_transform(X_test1)
X_test=sc.fit_transform(X_test)

pca = PCA(n_components=2)
X_train=pca.fit_transform(X_train,y_train)
X_test=pca.transform(X_test)

#print(X_train)

clf = LogisticRegression()
clf.fit(X_train,y_train)
accuracy =clf.score(X_test,y_test)

print(accuracy)

clf2 = LogisticRegression(max_iter=5000)
clf2.fit(X_train1,y_train)
acc2 = clf2.score(X_test1,y_test)

print(acc2)

print(X_train.T[0])


