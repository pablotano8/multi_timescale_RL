# import sys  
import argparse
import collections
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lib import wrappers

DEVICE = 'cuda'

DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 400_000 #200000
LEARNING_RATE = 1e-4 #1e-4
SYNC_TARGET_FRAMES = 10000
REPLAY_START_SIZE = 50_000 #200000

#EPSILON_DECAY_LAST_FRAME =1000000
#EPSILON_START = 1
#EPSILON_FINAL = 0.01
eps_initial=1
eps_final=0.1
eps_final_frame=0.01
eps_evaluation=0.0
eps_annealing_frames=1000000
max_frames=2000000


NUM_GAMMAS=1
taus=np.linspace(3,100,NUM_GAMMAS)
GAMMAS=np.exp(-1/taus)
GAMMAS[-1]=0.99

GAMMAS=np.flip(GAMMAS)


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

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
            nn.Linear(512, n_actions)))
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

    def fill_buffer(self,net,env,size):
        state = env.reset()

        while len(self.buffer)<size:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(DEVICE)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v.narrow(-1,0,env.action_space.n), dim=1)
            action = int(act_v.item())

            # do step in the environment
            new_state, reward, is_done, _ = env.step(action)

            exp = Experience(state, action, reward,
                            is_done, new_state)
            self.buffer.append(exp)
            state = new_state
            if is_done:
                state = env.reset()

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



def calc_loss(batch, net, tgt_net, device="cpu"):
    
    states, actions, rewards, dones, next_states = batch


    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)

    actions_v0 = torch.tensor(actions).to(device)
    output=net(states_v)
    output=output.reshape(len(rewards),NUM_GAMMAS,env.action_space.n)
    state_action_values=output[range(0,len(rewards)),:,actions_v0]

    
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    rewards_v=rewards_v.repeat(NUM_GAMMAS,1).T
    done_mask=done_mask.repeat(NUM_GAMMAS,1).T

    gammas=np.tile(GAMMAS,(len(rewards),1))
    gammas=np.float32(gammas)
    gammas=torch.tensor(gammas).to(device)

    rewards_v[done_mask] = -1

    with torch.no_grad():
        output_next=tgt_net(next_states_v)
        output_next=output_next.reshape(len(rewards),NUM_GAMMAS,env.action_space.n)

        best_action=torch.argmax(output_next[:,0,:],dim=-1)

        next_state_values=output_next[range(0,len(rewards)),:,best_action]
        next_state_values[done_mask] = 0.0

        next_state_values = next_state_values.detach()


    expected_state_action_values=next_state_values * gammas + \
                                   rewards_v

    return nn.SmoothL1Loss()(state_action_values,expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args,unknown = parser.parse_known_args()
    device = "cuda"

    env = wrappers.make_env(args.env)

    net = DQN2(env.observation_space.shape,
                        env.action_space.n).to('cpu')
    net.to('cuda')
    
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    net.apply(init_weights)

    with torch.no_grad():
        tgt_net = DQN2(env.observation_space.shape,
                                env.action_space.n).to('cpu')

    tgt_net.to('cuda')

    print(net)
    print(GAMMAS)

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
while True:
        frame_idx += 1

        if frame_idx < REPLAY_START_SIZE:
            epsilon = eps_initial
        elif frame_idx >= REPLAY_START_SIZE and frame_idx < REPLAY_START_SIZE + eps_annealing_frames:
            epsilon =  slope*frame_idx +  intercept
        elif frame_idx >=  REPLAY_START_SIZE +  eps_annealing_frames:
            epsilon =  slope_2*frame_idx + intercept_2

        if epsilon<eps_final_frame:
            epsilon=eps_final_frame

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

        
        
        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            with torch.no_grad():
                del tgt_net #LOAD_STATE_DICT causes memory issues (only seen when running "dmesg" in terminal)
                
                tgt_net = DQN2(env.observation_space.shape,
                            env.action_space.n).to('cpu')
                tgt_net.load_state_dict(net.state_dict())
                tgt_net.cuda()
                

        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        del loss_t
        del batch
 #   writer.close()


import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

mg1 = savgol_filter(np.loadtxt('rewards_mh_breakout.txt',dtype=int),705,1)
mg2 = savgol_filter(np.loadtxt('rewards_mh_breakout_2.txt',dtype=int),705,1)
mg3 = savgol_filter(np.loadtxt('rewards_mh_breakout_3.txt',dtype=int),705,1)
mg4 = savgol_filter(np.loadtxt('rewards_mh_breakout_4.txt',dtype=int),705,1)
mg_50 = savgol_filter(np.loadtxt('rewards_oh_50g_breakout.txt',dtype=int),705,1)
mg_20 = savgol_filter(np.loadtxt('rewards_mh_20g_breakout.txt',dtype=int),705,1)

# with open("rewards_breakout_1g_3.pkl", "wb") as output_file:
#     pickle.dump(total_rewards, output_file)
# torch.save(net, 'net_breakout_1g_2.pt')

with open("rewards_breakout_10g.pkl", "rb") as input_file:
    r = pickle.load(input_file)
mg5 = savgol_filter(r,705,1)

with open("rewards_breakout_10g_2.pkl", "rb") as input_file:
    r = pickle.load(input_file)
mg6 = savgol_filter(r,705,1)

sg3 = savgol_filter(np.loadtxt('rewards_breakout_3.txt',dtype=int),705,1)
sg4 =savgol_filter( np.loadtxt('rewards_breakout_4.txt',dtype=int),705,1)

with open("rewards_breakout_1g.pkl", "rb") as input_file:
    r = pickle.load(input_file)
sg5 = savgol_filter(r,705,1)
with open("rewards_breakout_1g_2.pkl", "rb") as input_file:
    r = pickle.load(input_file)
sg6 = savgol_filter(r,705,1)

with open("rewards_breakout_1g_3.pkl", "rb") as input_file:
    r = pickle.load(input_file)
sg7 = savgol_filter(r,705,1)

# plt.plot(savgol_filter(mg1,305,1),'r')
# plt.plot(savgol_filter(mg2,305,1),'r')
# plt.plot(savgol_filter(mg3,305,1),'r')
plt.plot(savgol_filter(mg5,305,1),'r')


# plt.plot(savgol_filter(mg_50,305,1),'r')
plt.plot(savgol_filter(mg6,305,1),'r')
# plt.plot(savgol_filter(mg_20,305,1),'r')
# plt.plot(np.mean([mg1[0:30000],mg2[0:30000],mg3[0:30000], mg4[0:30000]],axis=0),'r')
# plt.plot(np.mean([sg3[0:30000],sg4[0:30000]],axis=0),'k')
# 
# plt.plot(savgol_filter(sg3,305,1),'k')
# plt.plot(savgol_filter(sg4,305,1),'k')
# plt.plot(savgol_filter(sg5,305,1),'k')
# plt.plot(savgol_filter(sg6,305,1),'k')
plt.plot(savgol_filter(sg7,305,1),'k')
plt.grid(b=True)
plt.show()

