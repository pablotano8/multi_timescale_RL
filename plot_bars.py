
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

data = {
    'exp':[0,0,0,1,1,1,2,2,2],
    'gamma':[3,2,1,3,2,1,3,2,1],
    'mean':[0.99,0.9,0.33,0.99,0.98,0.76,1.0,0.98,0.8],
    }

data = {
    'exp':[0,0],
    'gamma':[1,50],
    'mean':[0.24,0.41],
    }

data = pd.DataFrame(data=data)


# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    data=data, x="gamma", y="mean", col="exp",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.ylim([0,0.5])
plt.show()

plt.plot()