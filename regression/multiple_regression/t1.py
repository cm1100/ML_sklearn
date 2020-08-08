import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df = pd.read_csv("50_Startups.csv")
#print(df.head())

#print(list(df))
df1 = np.array(df)
X = np.array(df.iloc[:,:-1])
y = np.array(df.iloc[:,-1])
y = y.reshape(y.shape[0],1)

#print(X.shape,y.shape,df1.shape)


X=X.T
print(X[3].shape)
lb = LabelEncoder()
X[3] = lb.fit_transform(X[3])

X=X.T
#print(X)
#print(X)

st = SimpleImputer(missing_values=np.nan,strategy="mean")
st.fit(X[:,0:3])
X[:, 0:3] = st.transform(X[:,0:3])

#print(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


st1 = StandardScaler()
X_train[:,:-1]=st1.fit_transform(X_train[:,:-1])
X_test[:,:-1]=st1.fit_transform(X_test[:,:-1])

#print(X_train)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

print(clf.predict(X_test))

print(list(df))

X_plot = X_train.T
print(X_plot[0])
plt.scatter(X_test.T[3],y_test,color="r")


predictions = clf.predict(X_test)

plt.scatter(X_test.T[3],predictions,color="k")
plt.show()
