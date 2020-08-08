import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("Wine.csv")

#print(list(df))
#print(df.head())

X = df.iloc[:,:-1].values
y= df.iloc[:,-1].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

X_train1,X_test1= X_train,X_test

kpca = KernelPCA(n_components=2,kernel="rbf")
kpca.fit_transform(X_train,y_train)
kpca.transform(X_test)


clf1 = LogisticRegression()
clf1.fit(X_train,y_train)

acc1 = clf1.score(X_test,y_test)
print(acc1)

clf2 = LogisticRegression()
clf2.fit(X_train1,y_train)

acc2 = clf2.score(X_test1,y_test)
print(acc2)
