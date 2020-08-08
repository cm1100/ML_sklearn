import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("Position_Salaries.csv")

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])

y = y.reshape(y.shape[0],1)

print(X.shape,y.shape)

clf = DecisionTreeRegressor(random_state=0)
clf.fit(X,y)
pred = clf.predict(X)
new = clf.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))


plt.plot(X_grid,clf.predict(X_grid),color="r")
plt.scatter(X,y,color="k")
plt.scatter([[6.5]],new,color="b")
plt.show()
