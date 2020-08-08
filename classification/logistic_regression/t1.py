import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import time

a = time.time()



df = pd.read_csv("Social_Network_Ads.csv")

print(df.head())

X = df.iloc[:,:-1].values
y= df.iloc[:,-1].values

print(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train.shape,y_train.shape)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)

clf = LogisticRegression()
clf.fit(X_train,y_train)

pred = clf.predict(sc.fit_transform(X_test))
print(pred)

print(y_test)

accuracy = clf.score(sc.fit_transform(X_test),y_test)
print(accuracy)
#print(X_test.T[0].shape,y_test.shape)

mat = confusion_matrix(y_test,pred)
print(mat)


# visualizing


X_set , y_set = sc.inverse_transform(X_train),y_train

#print(X_set[:,0])

X1,X2 = np.meshgrid(np.arange(start=min(X_set[:,0])-10,stop=max(X_set[:,0])+10,step=0.25),
                    np.arange(start=X_set[:,1].min()-1000,stop=X_set[:,1].max()+1000,step=0.25))

print(X1.shape,X2.shape)

plt.contourf(X1,X2,clf.predict(sc.transform(np.array([X1.ravel(),X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    print(i,j)
    plt.scatter(X_set[y_set==j,0],X_set[y_set==i,1],c=ListedColormap(("red","green"))(i),label=j)



plt.show()
b=time.time()

print(str(b-a))



