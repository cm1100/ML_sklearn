import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,accuracy_score




df = pd.read_csv("Churn_Modelling.csv")

print(df.head())
X= df.iloc[:,3:-1].values
y= df.iloc[:,-1].values

print(X.shape,y.shape)


lb = LabelEncoder()
lb1=LabelEncoder()
X.T[1]=lb.fit_transform(X.T[1])
X.T[2]=lb1.fit_transform(X.T[2])



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


ann = Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=4,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train,y_train,batch_size=32,epochs=150)

predict = ann.predict(X_test)
print(predict)

for i in range(len(predict)):
    if predict[i] <=0.5:
        predict[i]=0
    else:
        predict[i]=1

right=0

predict1 = np.round(predict)

print(predict)
for i in range(len(y_test)):
    if y_test[i]==predict1[i]:
        right+=1

accur = right/len(y_test)
print(accur)

accuracy2 = accuracy_score(y_test,predict1)
print(accuracy2)

mat = confusion_matrix(y_test,predict1)
print(mat)





