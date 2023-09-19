# import sys  
import argparse
import collections
import os
import time
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
import seaborn as sns


NUM_TRAINED_NETS = 20
DEFAULT_ENV_NAME = "LunarLander-v2"
NUM_FRAMES_PER_EXPERIMENT = 50_000

BATCH_SIZE = 32
REPLAY_SIZE = 200000 #15000
LEARNING_RATE = 1e-3 #1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000 #200000

EPSILON_DECAY_LAST_FRAME = 40000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


NUM_GAMMAS=25
taus=np.linspace(0,100,NUM_GAMMAS)
GAMMAS=np.exp(-1/taus)
GAMMAS[-1]=0.99


GAMMAS=np.flip(GAMMAS)


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])



class DQN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Sequential( 
            nn.Linear(state_dim, 512),
        )
        
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        self.fc3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, NUM_GAMMAS*action_dim),
        )
        
        
    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
        # return self.fc(x)
    
    
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
    def play_step(self, net, epsilon=0.0, device="cpu",learning=True):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # print(self.state)
            state_a = np.array([self.state], copy=False)
            # print(state_a)
            state_v = torch.tensor(state_a).to(device)
            # print(state_v)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v.narrow(-1,0,env.action_space.n), dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _= self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        if learning:
            self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward



def calc_loss(batch, net, tgt_net, device="cpu",individual=False):
    
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

    if individual==False:
        return nn.SmoothL1Loss()(state_action_values,expected_state_action_values)
    
    else:
        return nn.SmoothL1Loss(reduction='none')(state_action_values,expected_state_action_values)


if __name__ == "__main__":
    all_dif_high,all_dif_low = [],[]

    for _ in range(NUM_TRAINED_NETS):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda", default=False,
                            action="store_true", help="Enable cuda")
        parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                            help="Name of the environment, default=" +
                                DEFAULT_ENV_NAME)
        args,unknown = parser.parse_known_args()
        device = "cpu"

        env = gym.make(args.env)

        net = DQN(env.observation_space.shape[0],
                            env.action_space.n).to('cpu')
        
        with torch.no_grad():
            tgt_net = DQN(env.observation_space.shape[0],
                                    env.action_space.n).to('cpu')

        print(net)
        print(GAMMAS)

        buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Agent(env, buffer)
        epsilon = EPSILON_START

        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        total_rewards = []
        similarities=[]
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_m_reward = None

        while frame_idx<NUM_FRAMES_PER_EXPERIMENT:
            frame_idx += 1

            epsilon = max(EPSILON_FINAL, EPSILON_START -
                        frame_idx / EPSILON_DECAY_LAST_FRAME)

            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                m_reward = np.mean(total_rewards[-100:])
                
                if len(total_rewards) % 10 ==0:
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
                    
                    tgt_net = DQN(env.observation_space.shape[0],
                                env.action_space.n).to('cpu')
                    tgt_net.load_state_dict(net.state_dict())
                    
            
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()
            del loss_t
            del batch

        torch.save(net, f'lunar_lander.pt')

        device='cpu'
        agent.env.reset()

        tgt_net=net

        torch.save(net.state_dict(), 'net_cache')

        old_net = DQN(env.observation_space.shape[0],
                                    env.action_space.n).to(device)

        weights=[]; unnorm_weights=[]; pos=[]; rewards=[]; height= []
        for it in range(300):
            if it%50==0:
                print(it)
            agent.play_step(net, epsilon, device=device,learning=False)
            
            BATCH_SIZE=32
            start_index = np.random.choice(range(0,len(buffer)-BATCH_SIZE))
            batch_ind = buffer.ordered_sample(start_index,start_index+BATCH_SIZE)
            
            actions=batch_ind[1]; states=batch_ind[0]; reward=batch_ind[2]
            rewards.append(reward)

            states_v = torch.tensor(np.float32(np.array(
                states, copy=False))).to(device)

            output_old=net(states_v)
            old_q_vals=(output_old[range(0,len(actions)),actions]).detach().to('cpu').numpy()

            ratio, delta =[], []
            for gamma in range(0,len(GAMMAS)):
                optimizer_aux=torch.optim.Adam(old_net.parameters(), lr=LEARNING_RATE)
                optimizer_aux.zero_grad()
                
                old_net.load_state_dict(net.state_dict()); 
                old_net.zero_grad()
                    
                loss = calc_loss(batch_ind, old_net, net, device=device,individual=True)
                ind_loss=loss.mean(0)[gamma]
                ind_loss.backward(retain_graph=True)

                optimizer_aux.step()
                

                output=old_net(states_v).reshape(len(actions),NUM_GAMMAS,env.action_space.n)
                
                new_q_vals=torch.max(output[range(0,len(actions)),0:1,0:env.action_space.n],2).values.detach().to('cpu').numpy()
                

                delta.append(np.array(new_q_vals)-np.array(old_q_vals))

                ratio.append(np.array((np.sum(delta[gamma])/BATCH_SIZE)))

            ratio=ratio-np.mean(ratio)
            weights.append(ratio)
            height.append(np.mean(states,0)[1])


        dif_high = [w for i,w in enumerate(weights) if height[i]>np.median(height)]
        dif_low = [w for i,w in enumerate(weights) if height[i]<np.median(height)]

        # plt.figure
        # plt.plot(height)
        # plt.show()


        sns.set_theme(style='white')
        plt.figure(figsize=(2.3, 2.3))
        plt.plot([0,100],[0,0],color=[0.6,0.6,0.6])
        mean_low = savgol_filter(np.flip(np.mean(dif_low,0)),5,1)
        # mean_low = (mean_low-np.min(mean_low))/(np.max(mean_low-np.min(mean_low)))
        std_low = np.flip(np.std(dif_low,0))

        mean_high = savgol_filter(np.flip(np.mean(dif_high,0)),5,1)
        # mean_high = (mean_high-np.min(mean_high))/(np.max(mean_high-np.min(mean_high)))
        std_high = np.flip(np.std(dif_high,0))



        plt.plot(taus, mean_low, '-', label='mean_1')
        plt.fill_between(taus, mean_low - std_low/2, mean_low + std_low/2, color='C0', alpha=0.2)
        plt.plot(taus, mean_high, '-', label='mean_2')
        plt.fill_between(taus, mean_high - std_high/2, mean_high + std_high/2, color='C1', alpha=0.2)
        plt.plot(taus,mean_low,'.',color='C0')
        plt.plot(taus,mean_high,'.',color='C1')
        plt.yticks([0])
        # plt.savefig("lunarlander_adv.svg", format='svg')
        plt.show()

        all_dif_high.append(dif_high)
        all_dif_low.append(dif_low)