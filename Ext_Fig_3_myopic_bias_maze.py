import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

def move(size,pos_x,pos_y,act_x,act_y):
    return max((min((pos_x+act_x,size)),0)), max((min((pos_y+act_y,size)),0))


print('Computing True value function...')
# Generate true value dunction
size=10
pos_x , pos_y = 0 , 0
V=np.zeros((size+1,size+1))
alpha=0.05
gamma=0.99
num_episodes=200000
length_trajectory=100
for it in range(num_episodes):
    if it%10000==0:
        print(it)
    pos_x , pos_y = np.random.choice(range(0,size),2)
    count=0
    while True:
        count+=1
        if pos_x==size-1 and pos_y==size-9:
            r=np.random.normal(2,0.01)
        elif pos_x==size-1 and pos_y==size-5:
            r=np.random.normal(-4,0.01)
        elif pos_x==size-1 and pos_y==size-1:
            r=np.random.normal(2,0.01)           
        else:
            r=np.random.normal(0,0.01)
#             r=0
            
        if count<length_trajectory:
            act_x , act_y = np.random.choice([-1,0,1],2)
            pos_x_next , pos_y_next = move(size,pos_x , pos_y , act_x , act_y)
            V[pos_x,pos_y]=V[pos_x,pos_y] + alpha*(r + gamma*V[pos_x_next,pos_y_next] - V[pos_x,pos_y])
            pos_x , pos_y = pos_x_next , pos_y_next
        else:
            V[pos_x,pos_y]=V[pos_x,pos_y] + alpha*(r - V[pos_x,pos_y])
            break

    V_correct=V


print('Done')

print('Generating Trajectories...')
# Generate Trajectories
num_episodes=10000

POS, POS_NEXT, R, TER =[] , [] , [] , []
for _ in range(num_episodes):
    count=0
    positions, positions_next, rewards, termination = [], [], [], []
    while True:
        count+=1
        if pos_x==size-1 and pos_y==size-9:
            r=np.random.normal(2,0.01)
        elif pos_x==size-1 and pos_y==size-5:
            r=np.random.normal(-4,0.01)
        elif pos_x==size-1 and pos_y==size-1:
            r=np.random.normal(2,0.01)           
        else:
            r=np.random.normal(0,0.01)
#             r=0
            
        if count<length_trajectory:
            act_x , act_y = np.random.choice([-1,0,1],2)
            pos_x_next , pos_y_next = move(size,pos_x , pos_y , act_x , act_y)
            
            positions.append([pos_x,pos_y])
            pos_x , pos_y = pos_x_next , pos_y_next
            positions_next.append([pos_x_next,pos_y_next])
            rewards.append(r)
            termination.append(0)
        else:
            termination.append(1)
            pos_x , pos_y = np.random.choice(range(0,size),2)
            positions.append([pos_x,pos_y])
            pos_x_next , pos_y_next = move(size,pos_x , pos_y , act_x , act_y)
            pos_x , pos_y = pos_x_next , pos_y_next
            rewards.append(r)
            positions_next.append([pos_x_next,pos_y_next])
            break
    POS.append(positions)
    POS_NEXT.append(positions_next)
    R.append(rewards)
    TER.append(termination)
print('Done')

print('Evaluating gammas...')
# EVALUATE
import scipy.stats as stats

pos_x , pos_y = 0 , 0
alpha=0.1

# NUM_GAMMAS = 15
# GAMMAS = np.linspace(0.7,0.99,NUM_GAMMAS)

NUM_GAMMAS=25
GAMMAS=np.linspace(0.6,0.99,NUM_GAMMAS)
NUMBER_OF_TRAJECTORIES = 50


errors1=[]; errors2=[]; errors=[]
for it in range(300):
    print(it)
    V=np.zeros((size+1,size+1,NUM_GAMMAS))
    which_trajectories=np.random.choice(range(10000),NUMBER_OF_TRAJECTORIES)

    for i in range(0,NUM_GAMMAS):
        for _ in range(10000):
            which_trajectory=np.random.choice(which_trajectories)
            sample=np.random.choice(range(length_trajectory))
            
            ter = TER[which_trajectory][sample]
            pos_x , pos_y = POS[which_trajectory][sample]
            pos_x_next , pos_y_next = POS_NEXT[which_trajectory][sample]
            r = R[which_trajectory][sample]
            
            if ter==0:
                V[pos_x,pos_y,i] = V[pos_x,pos_y,i] + alpha*(r + GAMMAS[i]*V[pos_x_next,pos_y_next,i] - V[pos_x,pos_y,i])
            else:
                V[pos_x,pos_y,i] = V[pos_x,pos_y,i] + alpha*(r - V[pos_x,pos_y,i])

                
    error, error1, error2 = [], [] , []
    for i in range(0,NUM_GAMMAS):
        tau, _ = stats.kendalltau(V_correct,V[:,:,i])
        error.append(tau)
#         error.append(np.mean((V_correct-V[:,:,i])**2))
        tau, _ = stats.kendalltau(V_correct[0:5,:],V[0:5,:,i])
        error1.append(tau)
#         error1.append(np.mean((V_correct[0:5,:]-V[0:5,:,i])**2))
        tau, _ = stats.kendalltau(V_correct[6:,:],V[6:,:,i])
        error2.append(tau)
#         error2.append(np.mean((V_correct[6:,:]-V[6:,:,i])**2))
        
    errors.append(error)
    errors1.append(error1)
    errors2.append(error2)
print('Done')


sns.set_theme(style='white')
matlab_blue = (0, 0.2, 0.6)
matlab_red = (0.7, 0.1, 0)
norm_errors = (errors2-np.min(np.nanmean(errors2,0)))/(np.max(np.nanmean(errors2,0))-np.min(np.nanmean(errors2,0)))
mean2 = savgol_filter(np.nanmean(norm_errors,0),7,1)
# mean_low = (mean_low-np.min(mean_low))/(np.max(mean_low-np.min(mean_low)))
std2 = np.std(mean2,0)

norm_errors = (errors1-np.min(np.nanmean(errors1,0)))/(np.max(np.nanmean(errors1,0))-np.min(np.nanmean(errors1,0)))
mean1 = savgol_filter(np.nanmean(norm_errors,0),7,1)
# mean_low = (mean_low-np.min(mean_low))/(np.max(mean_low-np.min(mean_low)))
std1 = np.std(mean1,0)

fig, ax = plt.subplots(figsize=(4, 3))
plt.plot(GAMMAS, mean1, '-', label='mean_1',color=matlab_red)
plt.fill_between(GAMMAS, mean1 - std1/2, mean1 + std1/2, color=matlab_red, alpha=0.2)
plt.plot(GAMMAS, mean2, '-', label='mean_2',color=matlab_blue)
plt.fill_between(GAMMAS, mean2 - std2/2, mean2 + std2/2, color=matlab_blue, alpha=0.2)
plt.plot(GAMMAS,mean2,'.',color=matlab_blue)
plt.plot(GAMMAS,mean1,'.',color=matlab_red)
plt.show()

fig.savefig('figures/my_figure.svg', format='svg')

# plt.figure(dpi=150)
# norm = (np.max(np.nanmean(errors2,0))-np.min(np.nanmean(errors2,0)))
# plt.plot(GAMMAS,(np.nanmean(errors2,0)-np.min(np.nanmean(errors2,0)))/norm,'r')
# plt.plot(GAMMAS,(np.nanmean(errors2,0)-np.min(np.nanmean(errors2,0)))/norm,'ro')
# norm = (np.max(np.nanmean(errors1,0))-np.min(np.nanmean(errors1,0)))
# plt.plot(GAMMAS,(np.nanmean(errors1,0)-np.min(np.nanmean(errors1,0)))/norm,'b')
# plt.plot(GAMMAS,(np.nanmean(errors1,0)-np.min(np.nanmean(errors1,0)))/norm,'bo')
# norm = (np.max(np.nanmean(errors,0))-np.min(np.nanmean(errors,0)))
# plt.plot(GAMMAS,(np.nanmean(errors,0)-np.min(np.nanmean(errors,0)))/norm,'k')
# plt.plot(GAMMAS,(np.nanmean(errors,0)-np.min(np.nanmean(errors,0)))/norm,'ko')
# plt.show()


# # Results (I already ran them)
# GAMMAS = np.array([0.7       , 0.72071429, 0.74142857, 0.76214286, 0.78285714,
#        0.80357143, 0.82428571, 0.845     , 0.86571429, 0.88642857,
#        0.90714286, 0.92785714, 0.94857143, 0.96928571, 0.99      ])

# y1 = np.array([-0.05433766,  0.25635305,  0.56704375,  0.92518429,  1.23337362,
#         1.57363478,  1.87825319,  2.23851651,  2.54411493,  2.88545775,
#         3.08196232,  3.24264565,  3.28695128,  3.24635677,  3.20576227])

# y2 = np.array([0.74466098, 0.76295951, 0.78125803, 0.79851108, 0.7613913 ,
#        0.79291785, 0.75735657, 0.78217872, 0.74968564, 0.68094392,
#        0.60882914, 0.56520384, 0.36520384, 0.23690201, 0.10860017])

# plt.figure(dpi=150)
# plt.plot(GAMMAS,y1,color='C1')
# plt.plot(GAMMAS,y1,'.',color='C1')
# plt.plot(GAMMAS,y2,'C0')
# plt.plot(GAMMAS,y2,'.',color='C0')
# plt.show()


# Values for interesting trajectory
interesting_traj=[[8, 1],[8, 2],[8, 3],[8, 4],[8, 5],[9, 5],[9, 4],[9, 3],[9, 2],[9, 1]]
interesting_traj_next=[[8, 2],[8, 3],[8, 4],[8, 5],[9, 5],[9, 4],[9, 3],[9, 2],[9, 1],[9,1]]
# interesting_traj=[[1, 5],[2, 5],[3, 5],[4, 5],[6, 5],[7, 5],[8, 5],[9, 5],[9, 6],[9, 7]]
# interesting_traj_next=[[2, 5],[3, 5],[4, 5],[6, 5],[7, 5],[8, 5],[9, 5],[9, 6],[9, 7],[9, 7]]
rewards_interesting_traj = [-0.01,0.02,-0.02,0.05,0.05,-4,0.02,-0.001,0.0001,2]


V=np.zeros((size+1,size+1,NUM_GAMMAS))
which_trajectories=np.random.choice(range(10000),2)

for i in range(0,NUM_GAMMAS):
    for _ in range(10000):
        sample=np.random.choice(range(len(interesting_traj)))
        
        ter = interesting_traj[sample]==[9, 1]
        pos_x , pos_y = interesting_traj[sample]
        pos_x_next , pos_y_next = interesting_traj_next[sample]
        r = rewards_interesting_traj[sample]
        
        if ter==0:
            V[pos_x,pos_y,i] = V[pos_x,pos_y,i] + alpha*(r + GAMMAS[i]*V[pos_x_next,pos_y_next,i] - V[pos_x,pos_y,i])
        else:
            V[pos_x,pos_y,i] = V[pos_x,pos_y,i] + alpha*(r - V[pos_x,pos_y,i])

# plot comparison with true values
low_gamma_estimate,high_gamma_estimate,true_value = [],[],[]
for position in interesting_traj:
    true_value.append(V_correct[position[0],position[1]])
    low_gamma_estimate.append(V[position[0],position[1],0])
    high_gamma_estimate.append(V[position[0],position[1],-1])

fig = plt.figure(figsize=[4,3])
plt.plot([1,2,3,4,5,6,7,8,9,10],true_value,color=[0.4,0.4,0.4])
plt.plot([1,2,3,4,5,6,7,8,9,10],true_value,'.',color=[0.4,0.4,0.4])
plt.plot([1,2,3,4,5,6,7,8,9,10],high_gamma_estimate,color=[0.6,0.4,0.2])
plt.plot([1,2,3,4,5,6,7,8,9,10],high_gamma_estimate,'.',color=[0.6,0.4,0.2])
plt.plot([1,2,3,4,5,6,7,8,9,10],low_gamma_estimate,color=[0.4,0.8,0.4])
plt.plot([1,2,3,4,5,6,7,8,9,10],low_gamma_estimate,'.',color=[0.4,0.8,0.4])
plt.ylim([-8,5])
plt.show()

fig.savefig('figures/my_figure.svg', format='svg')

