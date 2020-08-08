import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


df = pd.read_csv("Data.csv")
print(list(df))

X = np.array(df.iloc[:,:-1])
y = np.array(df.iloc[:,-1])

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)



print(X_train.shape)

clf = XGBClassifier()
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)
print(acc)

accuracies =cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10)
print(accuracies)