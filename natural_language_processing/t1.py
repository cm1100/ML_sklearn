import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

df = pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)#qouting used for ignoring quates


corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    pt = PorterStemmer()
    all_stop_words = stopwords.words('english')
    all_stop_words.remove('not')
    review =[pt.stem(i) for i in review if not i in set(all_stop_words)]
    review= ' '.join(review)
    corpus.append(review)

print(corpus)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y= np.array(df.iloc[:,-1])
print(X.shape)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

clf1 = GaussianNB()
clf1.fit(X_train,y_train)
accuracy =clf1.score(X_test,y_test)
pred1 = clf1.predict(X_test)
mat1= confusion_matrix(y_test,pred1)

clf2 = LogisticRegression()
clf2.fit(X_train,y_train)
accuracy1=clf2.score(X_test,y_test)
pred2 = clf2.predict(X_test)
mat2= confusion_matrix(y_test,pred2)
print(accuracy,accuracy1,"\n",mat1,"\n",mat2)

clf31= SVC(kernel='rbf')
clf31.fit(X_train,y_train)
accuracy3= clf31.score(X_test,y_test)
print(accuracy3)
