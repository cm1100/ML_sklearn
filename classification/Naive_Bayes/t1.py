import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("Social_Network_Ads.csv")

X = df.iloc[:,:-1]
y= df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

clf = GaussianNB()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

