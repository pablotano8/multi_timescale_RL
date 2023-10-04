# import sys  
import argparse
import collections
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_ENV_NAME = "LunarLander-v2"
NUM_EXPERIMENTS = 5
NUM_FRAMES_PER_EXPERIMENT = 125_000

BATCH_SIZE = 32
REPLAY_SIZE = 200000 #200000
LEARNING_RATE = 1e-3 #1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000 #200000

EPSILON_DECAY_LAST_FRAME = 40_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


NUM_GAMMAS=25
# taus=np.linspace(0,100,NUM_GAMMAS)
# GAMMAS=np.exp(-1/taus)
# GAMMAS[-1]=0.99

# GAMMAS =0.99 + 0* GAMMAS
GAMMAS = np.linspace(0.6,0.99,NUM_GAMMAS)
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
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
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

    reward_experiment = []
    for i in range(NUM_EXPERIMENTS):
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
    
        reward_experiment.append(total_rewards)
        torch.save(net, f'nets/lunar_multi_gamma_linear_expert_{i}.pt')


# import pickle

# with open('rewards_10g_lunar_lander.pkl', 'wb') as f:
#     pickle.dump(reward_experiment, f)

# with open('rewards_10g_lunar_lander.pkl', 'rb') as f:
#     rewards = pickle.load(f)

# min([len(rewards[i]) for i in range(len(rewards))])
# mean_reward = []
# for i in range(395):
#     mean_reward.append(np.mean([rewards[n][i] for n in range(len(rewards))]))

# import matplotlib.pyplot as plt

# plt.plot(mean_reward)
# plt.grid(b=True)
# plt.show()


# import pickle

# with open('rewards_breakout_10g.pkl', 'wb') as f:
#     pickle.dump(total_rewards, f)

# with open('rewards_breakout_10g.pkl', 'rb') as f:
#     rewards = pickle.load(f)

# import matplotlib.pyplot as plt

# plt.plot(rewards)
# plt.grid(b=True)
# plt.show()
