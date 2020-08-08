import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random,math


df = pd.read_csv("Ads_CTR_Optimisation.csv")
print(df.head())

N=500
d=10
ads_selected = []
number_of_rewards_1=[0] * d
number_of_rewards_0=[0]*d
total_rewards=0

for n in range(N):
    ad =0
    max_random=0
    for i in range(d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
        if max_random<random_beta:
            max_random=random_beta
            ad =i
    ads_selected.append(ad)
    if df.values[n,ad]==0:
        number_of_rewards_0[ad]+=1
    else:
        number_of_rewards_1[ad]+=1
    total_rewards+=df.values[n,ad]


print(ads_selected)

plt.hist(ads_selected,rwidth=0.8)
plt.show()
