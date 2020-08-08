import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

df = pd.read_csv("Market_Basket_Optimisation.csv",header=None)

print(len(list(df)))
'''transactoins=[]

for i in range(0,7501):
    transactoins.append([str(df.values[i,j]) for j in range(0,20)])
'''
X=np.array(df,dtype='str')

#print(X)

rules = apriori(transactions=X,min_support=0.003,min_confidence=0.2,min_lift=3,min_lenght=2,max_length=2)


res = list(rules)
lhs = []
rhs=[]
support = []
confidence=[]
lift=[]
for i in res:
    lhs.append(tuple(i[0])[0])
    rhs.append(tuple(i[0])[1])
    support.append(i[1])
    confidence.append(tuple(i[2][0])[2])
    lift.append(tuple(i[2][0])[3])



#print(f"{lhs}\n{rhs}\n{support}\n{confidence}\n{lift}")
nlist = list(zip(lhs,rhs,support,confidence,lift))
df_n = pd.DataFrame(nlist,columns=["Side Deal","Real","Support","Confidence","Lift"])
#print(df_n.head())


print(df_n.nlargest(n=10,columns="Lift"))






