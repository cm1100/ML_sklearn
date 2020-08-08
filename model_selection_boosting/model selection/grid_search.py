import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



df = pd.read_csv("Social_Network_Ads.csv")

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

clf = SVC(kernel="rbf",)
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)

print(acc)


accuracies = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10)
print(accuracies.std())

parameters = [{'C':[0.1,0.25,0.5,0.75,1,1.25],'kernel':['linear']},
              {'C':[0.1,0.25,0.5,0.75,1,1.25],'kernel':['rbf','poly'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]}]


grid_search = GridSearchCV(estimator=clf,param_grid=parameters,scoring='accuracy',
                           cv=10,n_jobs=-1)
grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
print(best_accuracy)
parameters1= grid_search.best_params_
print(parameters1)




