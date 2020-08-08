import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

print(df.head(),list(df))

X = np.array(df.iloc[:,3:5])

print(X.shape)

wcss=[]
for i in range(1,11):
    clf = KMeans(n_clusters=i,init='k-means++',random_state=42)
    clf.fit(X)
    wcss.append(clf.inertia_)


#plt.plot(range(1,11),wcss)
#plt.show()



clf = KMeans(n_clusters=5,random_state=42,init='k-means++')
pred = clf.fit_predict(X)

#print(pred)

colors ={0:'r',1:'b',2:'g',3:'k',4:'y'}


for i in range(X.shape[0]):
    plt.scatter(X[i][0],X[i][1],color=colors[pred[i]])
    #print(colors[pred[i]])

for i in range(len(colors)):
    plt.scatter([],[],color=colors[i],label="scatter"+str(i))

plt.scatter(clf.cluster_centers_[:,0],clf.cluster_centers_[:,1],s=300,label="centroids")

plt.legend()

plt.show()
