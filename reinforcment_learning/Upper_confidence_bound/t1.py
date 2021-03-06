import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("Ads_CTR_Optimisation.csv")

print(len(df))

N=1000
d=10
X=np.array(df)
print(X.shape)
numbers_of_selections =[0] * d
sums_of_rewards = [0] *d
total_reward =0
ads_selected=[]

for n in range(N):
    ad =0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_selections[i]>0:
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/numbers_of_selections[i])
            upper_bound = average_reward+delta_i

        else:
            upper_bound=1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selections[ad]+=1
    sums_of_rewards[ad]+=df.values[n,ad]
    total_reward+=df.values[n,ad]

print(ads_selected)

plt.hist(ads_selected,rwidth=0.8)
plt.show()


        
