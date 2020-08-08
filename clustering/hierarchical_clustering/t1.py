import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering



df = pd.read_csv("Mall_Customers.csv")

X = np.array(df.iloc[:,3:5])

print(X[:5,:])

#dendogram = sch.dendrogram(sch.linkage(X,method='ward'))

#plt.title("dendogram")
#plt.xlabel('Costumers')
#plt.ylabel('Distances')
#plt.show()

clf = AgglomerativeClustering(n_clusters=3,affinity='euclidean')
pred = clf.fit_predict(X)

print(pred)

colors = {0:'r',1:'b',2:"k"}

#for i in range(X.shape[0]):
#    plt.scatter(X[i][0],X[i][1],color=colors[pred[i]])

#plt.show()

plt.scatter(X[pred==0,0],X[pred==0,1],color="r",label="first")
plt.scatter(X[pred==1,0],X[pred==1,1],color="k",label="second")
plt.scatter(X[pred==2,0],X[pred==2,1],color="magenta",label="third")

plt.show()



