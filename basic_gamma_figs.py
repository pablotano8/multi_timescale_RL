import matplotlib.pyplot as plt
import numpy as np
import matplotlib

gamma = np.flip(np.linspace(0.8,1,20))

cmap = matplotlib.cm.get_cmap('cool')

for i,reward_time in enumerate([0.5,1,40]):

    #responses to different reward sizes
    response = gamma**reward_time
    response2 = 1.5*gamma**reward_time


    if i==0:
        plt.plot(gamma,response,'-',color=cmap(1-i*0.95),linewidth=5)
        plt.plot(gamma,response,'o',color=[0.6,0.6,0.6],linewidth=0.5,alpha=0.5)
    else:
        plt.plot(gamma,2*(response-np.min(response))/(np.max(response)-np.min(response)),'-',color=cmap(1-i*0.95),linewidth=5)
        plt.plot(gamma,2*(response-np.min(response))/(np.max(response)-np.min(response)),'o',color=[0.6,0.6,0.6],linewidth=0.5,alpha=0.5)
    

    # if i==0:
    #    plt.plot(gamma[10],response[10],'o',color=cmap(1-i/3),linewidth=3)
    # else:
    #     plt.plot(gamma[10],(response[10]-np.min(response))/(np.max(response)-np.min(response)),'o',color=cmap(1-i/3),linewidth=3)

    plt.yticks([])
    plt.xticks([0.9])
plt.show()



#Plot temporal space
import matplotlib
from scipy.stats import norm
reward_times = [0,5,15,40]
cmap = matplotlib.cm.get_cmap('cool')
for i,reward_time in enumerate([0,5,15,40]):

    plt.figure(figsize=(3,1))
    #responses to different reward sizes
    x = np.linspace(-10,50,1000)

    # plt.plot(x,2*norm.pdf(x,reward_time,1+reward_time/10),color=cmap(1-i/3),linewidth=3)
    plt.plot(x,2*norm.pdf(x,reward_time,1),color=cmap(1-i/3),linewidth=3)
    
    #response to reward size 2
    plt.yticks([])
    plt.axis('off')
    plt.show()



#Exponential discountfrom scipy.stats import norm

#responses to different reward sizes
gammas = np.flip(np.linspace(0.1,1,20))

for gamma in gammas:
    x = np.linspace(0,10,1000)

    plt.plot(x,gamma**x,color=[0.6,0.6,0.6],linewidth=3,alpha=0.5)

    #response to reward size 2
    plt.yticks([])
    plt.axis('off')
plt.show()