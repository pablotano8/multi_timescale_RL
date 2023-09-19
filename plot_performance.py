import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter

sns.set_theme(style="whitegrid")

max_plot=40_000
min_plot=1
smooth = 750
skipped_points = 100

with open("rewards_breakout_10g.pkl", "rb") as input_file:
    mg1 = pickle.load(input_file)

with open("rewards_breakout_10g_2.pkl", "rb") as input_file:
    mg2 = pickle.load(input_file)

multi=[savgol_filter(np.loadtxt('rewards_oh_50g_breakout.txt',dtype='int'),smooth,1)[min_plot:max_plot:skipped_points],
        savgol_filter(mg1,smooth,1)[min_plot:max_plot:skipped_points],
          savgol_filter(mg2,smooth,1)[min_plot:max_plot:skipped_points]
         ]

x_multi=[np.linspace(0,max_plot/skipped_points,len(multi[i])) for i in range(0,len(multi))]

with open("rewards_breakout_1g.pkl", "rb") as input_file:
    sg1 = pickle.load(input_file)
with open("rewards_breakout_1g_2.pkl", "rb") as input_file:
    sg2 = pickle.load(input_file)


single=[ savgol_filter(np.loadtxt('rewards_breakout_4.txt',dtype='int'),smooth,1)[min_plot:max_plot:skipped_points],
         savgol_filter(sg1,smooth,1)[min_plot:max_plot:skipped_points],
         savgol_filter(sg2,smooth,1)[min_plot:max_plot:skipped_points],
        ]
x_single=[np.linspace(0,max_plot/skipped_points,len(single[i])) for i in range(0,len(single))]


df= pd.DataFrame({"multi" : np.concatenate(np.array(multi)),
                  "x_multi" : np.concatenate(np.array(x_multi)),
                 })
df2= pd.DataFrame({"single" : np.concatenate(np.array(single)),
                  "x_single" : np.concatenate(np.array(x_single)),
                 })

plt.figure(dpi=150)
print('Computing confidence intervals...')
fig=sns.lineplot(x="x_multi", y="multi", data=df,color='tab:red',ci=50)
print('Done 1')
fig=sns.lineplot(x="x_single", y="single", data=df2,color='k',ci=50)
print('Done 2')

plt.grid()
plt.savefig('demo.png', transparent=True)
plt.show() 
