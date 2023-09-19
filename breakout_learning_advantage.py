# import sys  
import os
#sys.path.insert(0, '/home/pablotano/Desktop/DeepRL/lib')

from gym import wrappers
from gym.wrappers import AtariPreprocessing, FrameStack

import torch.optim as optim
import argparse
import time
import numpy as np
import collections
import gym
import torch
import torch.nn as nn
import torch.optim as optim

# from tensorboardX import SummaryWriter

# torch.cuda.empty_cache()

DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 1000 #200000
LEARNING_RATE = 0.0000625 #1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 50000 #200000

#EPSILON_DECAY_LAST_FRAME =1000000
#EPSILON_START = 1
#EPSILON_FINAL = 0.01
eps_initial=0
eps_final=0.
eps_final_frame=0.01
eps_evaluation=0.0
eps_annealing_frames=1000000
max_frames=2000000

NUM_H=1
NUM_GAMMAS=50
ratio=np.ones(NUM_GAMMAS)
taus=np.linspace(1,100,NUM_GAMMAS)
GAMMAS=np.exp(-1/taus)
GAMMAS[-1]=0.99

GAMMAS=np.flip(GAMMAS)

GAMMAS=np.repeat(GAMMAS,NUM_H)

sens=[1,4,7]

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


def make_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
    env = FrameStack(env, num_stack=4)  # If you want to stack 4 frames together
    return env

class DQN2(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        
        self.layers = nn.ModuleList()
        for _ in range(NUM_GAMMAS):
            self.layers.append(nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions*NUM_H)))
          
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
    
        out = torch.cat([net(conv_out) for net in self.layers],1)
                 
        return out
    
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()#

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        
        self.layers = nn.ModuleList()
        for _ in range(NUM_GAMMAS):
            self.layers.append(nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions*NUM_H)))
          
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
    
        out = torch.cat([net(conv_out) for net in self.layers],1)
                 
        return out

    
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)
    
    def ordered_sample(self,start,end):
        indices = np.arange(start,end)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v.narrow(-1,0,env.action_space.n), dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward



def calc_loss(batch, net, tgt_net, device="cpu",individual=False,weights=None):
    
    states, actions, rewards, dones, next_states = batch


    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)

    actions_v0 = torch.tensor(actions).to(device)
    output=net(states_v)
    output=output.reshape(len(rewards),NUM_GAMMAS*NUM_H,env.action_space.n)
    state_action_values=output[range(0,len(rewards)),:,actions_v0]

    
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    rewards_v=rewards_v.repeat(NUM_GAMMAS*NUM_H,1).T
    done_mask=done_mask.repeat(NUM_GAMMAS*NUM_H,1).T

    gammas=np.tile(GAMMAS,(len(rewards),1))
    gammas=np.float32(gammas)
    gammas=torch.tensor(gammas).to(device)

    rewards_v[done_mask] = -1

    rewards_after_sens=rewards_v

    with torch.no_grad():
        output_next=tgt_net(next_states_v)
        output_next=output_next.reshape(len(rewards),NUM_GAMMAS*NUM_H,env.action_space.n)

        best_action=torch.argmax(output_next[:,0,:],dim=-1)

        next_state_values=output_next[range(0,len(rewards)),:,best_action]

        next_state_values[done_mask] = 0.0
    #     rewards_v[done_mask] = -1
        next_state_values = next_state_values.detach()


    expected_state_action_values=next_state_values * gammas + \
                                   rewards_after_sens


    
    losses=torch.transpose(nn.SmoothL1Loss(reduction='none')(state_action_values,expected_state_action_values),0,1)
    
    if individual==False:
#         return nn.SmoothL1Loss()(state_action_values,
#                     expected_state_action_values)
        return torch.mean(torch.mean(losses,1))
    
    else:
        return torch.mean(losses,1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args,unknown = parser.parse_known_args()
    device = "cpu"

    env = make_env(args.env)

    net = DQN(env.observation_space.shape,
                        env.action_space.n).to('cpu')
    net.to('cpu')
    
    with torch.no_grad():
        tgt_net = DQN(env.observation_space.shape,
                                env.action_space.n).to('cpu')

    tgt_net.to('cpu')

    net = torch.load(os.getcwd()+'/dqn_mh_breakout.pt')#
    # #net.load_state_dict(torch.load(os.getcwd()+'/PongNoFrameskip-v4-best_7.dat'))
    tgt_net = torch.load(os.getcwd()+'/dqn_mh_breakout.pt')
    
  #  writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = eps_initial
    slope = -( eps_initial -  eps_final)/ eps_annealing_frames
    intercept =  eps_initial -  slope* REPLAY_START_SIZE
    slope_2 = -( eps_final -  eps_final_frame)/( max_frames -  eps_annealing_frames -  REPLAY_START_SIZE)
    intercept_2 =  eps_final_frame -  slope_2* max_frames

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    similarities=[]
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

while frame_idx<2000:
    frame_idx += 1
    if frame_idx < REPLAY_START_SIZE:
        epsilon = eps_initial
    elif frame_idx >= REPLAY_START_SIZE and frame_idx < REPLAY_START_SIZE + eps_annealing_frames:
        epsilon =  slope*frame_idx +  intercept
    elif frame_idx >=  REPLAY_START_SIZE +  eps_annealing_frames:
        epsilon =  slope_2*frame_idx + intercept_2

    if epsilon<eps_final_frame:
        epsilon=eps_final_frame

#         epsilon=0.01
    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        m_reward = np.mean(total_rewards[-100:])
        
        if len(total_rewards) % 50 ==0:
            print("%d: done %d games, reward %.3f, "
                    "eps %.2f, speed %.2f f/s, rew epi: %.2f" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed, reward
            ))


import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
epsilon=0.0

device='cpu'
# env = wrappers.make_env(DEFAULT_ENV_NAME)
# env.reset()
# env = gym.make(args.env)
agent.env.reset()
# plt.figure(dpi=100)
tgt_net=net
weights=[]; y_pos=[]; rewards=[]
env.reset()
torch.save(net.state_dict(), 'net_cache')

old_net = DQN(env.observation_space.shape,
                            env.action_space.n).to('cpu')
net = DQN(env.observation_space.shape,
                            env.action_space.n).to('cpu')
net = torch.load(os.getcwd()+'/dqn_mh_breakout.pt')

for iteration in range(1000):
    if iteration%100==0:
        print(iteration)
    net = DQN(env.observation_space.shape,
                                env.action_space.n).to('cuda')
    net = torch.load(os.getcwd()+'/dqn_mh_breakout.pt')
    agent.play_step(net, epsilon, device=device)
    
    BATCH_SIZE=15
    batch_ind = buffer.ordered_sample(len(buffer)-BATCH_SIZE,len(buffer))

    
    actions=batch_ind[1]; states=batch_ind[0]; reward=batch_ind[2]

    states_v = torch.tensor(np.float32(np.array(
        states, copy=False))).to(device)

    output_old=net(states_v)
    old_q_vals=(output_old[range(0,len(actions)),actions]).detach().to('cpu').numpy()

    ratio=[]; delta=[]
    for gamma in range(0,NUM_GAMMAS-1):
        optimizer_aux=torch.optim.Adam(old_net.parameters(), lr=LEARNING_RATE)
        optimizer_aux.zero_grad()
        
        old_net.load_state_dict(net.state_dict()); 
        # old_net = torch.load(os.getcwd()+'/dqn_mh_breakout.pt')
        old_net.zero_grad()
            
        loss = calc_loss(batch_ind, old_net, net, device=device,individual=True)
        ind_loss=loss[gamma]
        ind_loss.backward(retain_graph=True)
        
       # with torch.no_grad():
       #      for p in old_net.parameters():
       #         new_value = p - LEARNING_RATE*p.grad
       #         p.copy_(new_value)
            
        optimizer_aux.step()

        output=old_net(states_v)
        
        new_q_vals=torch.max(output[range(0,len(actions)),0:env.action_space.n],1).values.detach().to('cpu').numpy()

        delta.append(np.array(new_q_vals)-np.array(old_q_vals))

        ratio.append(np.array((np.sum(delta[gamma])/BATCH_SIZE)))

    ratio=ratio
#   y_pos.append(np.mean((states_v[0][1]).detach.to('cpu').numpy()))
    weights.append(ratio)
    rewards.append(np.sum(reward))
    
    # plt.figure(dpi=250)
    # plt.subplot(1,3,1)
    # plt.plot(-2/np.log(GAMMAS[:-1]),np.array(ratio).flatten(),'k'); 
    # plt.plot(-2/np.log(GAMMAS[:-1]),np.array(ratio).flatten(),'ko'); 
    # plt.plot([-5,205],[0,0],color=[0.6,0.6,0.6]);  
    # plt.ylim([-0.1,0.1])
    # plt.subplot(1,3,2)
    # plt.title([np.mean(old_q_vals),np.mean(new_q_vals)])
    # plt.imshow(np.sum(np.sum((states_v).detach().to('cpu').numpy(),0),0),vmax=1);
    # if iteration>1:
    #     plt.subplot(1,3,3)
    #     plt.imshow(np.sum(np.sum((states_v).detach().to('cpu').numpy(),0),0)
    #                -np.sum(np.sum((old_states).detach().to('cpu').numpy(),0),0),vmin=0.05); 
    # old_states=states_v
    # plt.show()
 #   agent.env.render()


import seaborn as sns
import pandas as pd

down=[weights[i] for i,r in enumerate(rewards) if r>1]
x_ax=[-1/np.log(GAMMAS[:-1]) for i,r in enumerate(rewards) if r>1]

# df=pd.DataFrame(np.array(down)).melt()
# df= pd.DataFrame({"Weight" : np.array(down).flatten(),
#                   "Gamma" : np.array(x_ax).flatten()})
df = pd.read_pickle("./down.pkl")  

high=[weights[i] for i,r in enumerate(rewards) if r<1]
x_ax=[-1/np.log(GAMMAS[:-1]) for i,r in enumerate(rewards) if r<1]

# df=pd.DataFrame(np.array(down)).melt()
# df2= pd.DataFrame({"Weight" : np.array(high).flatten(),
#                   "Gamma" : np.array(x_ax).flatten()})
df2 = pd.read_pickle("./high.pkl")  
sns.set_theme(style="white")
plt.figure(dpi=100)
fig=sns.lineplot(x="Gamma", y="Weight", data=df,ci=65,color='b')
fig=sns.lineplot(x="Gamma", y="Weight", data=df2,ci=65,color='r')
# fig=sns.lineplot(x="Gamma", y="Weight", data=df2,ci=65,color='C1')
plt.plot([0,100],[0,0],color=[0.6,0.6,0.6])

means = np.array(df.groupby('Gamma')['Weight'].mean())
means2 = np.array(df2.groupby('Gamma')['Weight'].mean())
plt.plot(-1/np.log(GAMMAS[1:]),means2-np.mean(means2),'.',color='C1')
plt.plot(-1/np.log(GAMMAS[1:]),means2-np.mean(means2),color='C1')
plt.plot(-1/np.log(GAMMAS[1:]),means-np.mean(means),'.',color='C0')
plt.yticks([0])
plt.show()

#plt.plot(-2/np.log(GAMMAS),np.mean(np.array(down).T,1),'o',color='steelblue',markersize=4)
plt.savefig('demo.png', transparent=True)

