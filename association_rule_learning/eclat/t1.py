import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori


df = pd.read_csv("Market_Basket_Optimisation.csv")
X = np.array(df,dtype="str")


rules = apriori(transactions=X,min_support=0.003,min_confidence=0.2,min_lift=3,min_lenght=2,max_length=2)

res = list(rules)


support=[]
lhs=[]
rhs=[]

for i in res:
    lhs.append(tuple(i[0])[0])
    rhs.append(tuple(i[0])[1])
    support.append(i[1])
list_n = list(zip(lhs,rhs,support))

new_df = pd.DataFrame(list_n,columns=["lhs","rhs","support"])

print(new_df.nlargest(n=10,columns="support"))
