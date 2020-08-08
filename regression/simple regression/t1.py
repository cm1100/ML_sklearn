import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Salary_Data.csv")

X = df.iloc[:,:-1].values
y=df.iloc[:,-1].values

print(len(X))

#print(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

clf = LinearRegression()
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print(prediction)
print(y_test)


plt.scatter(X_train,y_train,c="k")
plt.plot(X_test,prediction,color="r")
plt.scatter(X_test,y_test,color="b")
plt.show()


accuracy = clf.score(X_test,y_test)
print(accuracy)

