import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data.csv")


X = df.iloc[:,:-1].values
print(X.shape)
y=df.iloc[:,-1]
print(y)

#missing data

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#print(X)

# handeling non numeric data

print(list(X[0][0][0]))

'''if list(X[0][0])=="France":
    print("yes")
else:
    print("no")'''

X_n = X.T
#print(X_n[0][0])
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#print(X)

lb = LabelEncoder()
y = lb.fit_transform(y)
#print(y)

lb1 = LabelEncoder()
X_n[0]=lb1.fit_transform(X_n[0])
#print(X_n[0])

#splitting data in train test and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
#print(X_train)
#print(X_test,"\n",y_train,"\n",y_test)


#feature scaling, standardization
sc = StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.fit_transform(X_test[:,3:])


#print(X_train)









