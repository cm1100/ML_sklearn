import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  PolynomialFeatures


df = pd.read_csv("Position_Salaries.csv")
#print(df)

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])

y= y.reshape(y.shape[0],1)

print(X.shape,y.shape)


clf1 = LinearRegression()
clf1.fit(X,y)
accuracy = clf1.score(X,y)
pred = clf1.predict(X)

plt.scatter(X,y,color="k")
plt.plot(X,pred,color="r")
#print(accuracy)

plt.show()



#polynomial regression

pr = PolynomialFeatures(degree=5)
X_new=pr.fit_transform(X)
clf2= LinearRegression()
clf2.fit(X_new,y)
accuracy1= clf2.score(X_new,y)
print(accuracy1)
pred2 =clf2.predict(X_new)




plt.scatter(X,y,color="r")
plt.plot(X,pred2,color="k")

plt.show()
print(X)

a=np.array([[6.5,1,2]])

print(a)

r=clf2.predict(pr.fit_transform([[6.6]]))
print(r)






