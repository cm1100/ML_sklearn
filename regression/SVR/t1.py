import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df = pd.read_csv("Position_Salaries.csv")

print(df)

X =np.array(df.iloc[:,1:-1])
y=np.array(df.iloc[:,-1])

y = y.reshape(y.shape[0],1)

sc = StandardScaler()
X_n  =sc.fit_transform(X)

sc1= StandardScaler()
sc1.fit(y)
y_n = sc1.fit_transform(y)

#print(X,y)

clf = SVR(kernel='rbf')
clf.fit(X_n,y_n)

predictions = sc1.inverse_transform(clf.predict(X_n))
print(predictions)

plt.scatter(X,y,color="r")
plt.plot(X,predictions,color="k")
plt.show()

