import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Position_Salaries.csv")
#print(df)

X = np.array(df.iloc[:,1:-1])
y = df.iloc[:,-1].values

#y= y.reshape(y.shape[0],1)

print(X.shape,y.shape)

clf = RandomForestRegressor(n_estimators=10,random_state=0)
clf.fit(X,y)

#pred = clf.predict(X)

X_grid = np.arange(min(X),max(X),0.01)

X_grid=X_grid.reshape(len(X_grid),1)
print(X_grid.shape)

n_pred = clf.predict(X_grid)
print(n_pred.shape)
n_pred = n_pred.reshape(n_pred.shape[0],1)

plt.scatter(X,y,color="k")
plt.plot(X_grid,n_pred,color="r")
plt.show()

