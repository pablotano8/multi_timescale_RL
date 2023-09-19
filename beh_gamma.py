# # !/usr/bin/env python3
import argparse
import collections
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# env = gym.make("LunarLander-v2")
# env.reset()
# h = []
# for i in range(5000):
#     print(i)
#     state, reward, is_done, _ = env.step(env.action_space.sample())
#     # env.render()
#     h.append(state[1])
#     # print(state[1])
#     if is_done:
#         env.reset()

# plt.plot(h,'.')
# plt.show()
# print(np.median(h))


device = 'cpu'
# 
DEFAULT_ENV_NAME = "LunarLander-v2"
MEAN_REWARD_BOUND = 19

MEDIAN_HEIGHT = 0.6
MAX_EXPERIENCE = 30000
REWARD_NOISE = 1

GAMMA_HIGH = 0.99
GAMMA_LOW = 0.99

BATCH_SIZE = 32
REPLAY_SIZE = 150000
LEARNING_RATE = 1e-3
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000


EPSILON_DECAY_LAST_FRAME = 30000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01



Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,env.action_space.n)
        )
        
    def forward(self, x):
        return self.fc(x)
    
    
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
    def play_step(self, net_low_gamma, net_high_gamma, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.float32(np.array([self.state], copy=False))

            state_v = torch.tensor(state_a).to(device)
            if state_a[0][1]>MEDIAN_HEIGHT:
                q_vals_v = net_high_gamma(state_v)
            else:
                q_vals_v = net_low_gamma(state_v)
            _, act_v = torch.max(q_vals_v.narrow(-1,0,env.action_space.n), dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # Add noise to rewards
        reward += np.random.normal(0,REWARD_NOISE)

        if frame_idx<MAX_EXPERIENCE:
            exp = Experience(self.state, action, reward,
                            is_done, new_state)
            self.exp_buffer.append(exp)

        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    
    states, actions, rewards, dones, next_states = batch

    # Create tensors from the variables in batch
    states_v = torch.tensor(np.float32(np.array(
        states, copy=False))).to(device)
    next_states_v = torch.tensor(np.float32(np.array(
        next_states, copy=False))).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    rewards_v[done_mask] = -1
    
    # Get the Q-values of the states(t) in batch, by applying our net
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    # Get the Q-values of the states(t+1) in batch, by applying our target_net
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    # TD target
    expected_state_action_values = next_state_values * gamma + \
                                   rewards_v
    
    # The Loss is the squared TD error
    return nn.SmoothL1Loss()(state_action_values,
                        expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args,unknown = parser.parse_known_args()
    device = "cpu"

    env = gym.make(args.env)

    net_low_gamma = DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    net_high_gamma = DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    with torch.no_grad():
        tgt_net_low_gamma = DQN(env.observation_space.shape,
                                env.action_space.n).to('cpu')
        tgt_net_high_gamma = DQN(env.observation_space.shape,
                                env.action_space.n).to('cpu')

    print(net_low_gamma)
    print('GAMMA LOW: ', GAMMA_LOW, ', GAMMA HIGH: ', GAMMA_HIGH, )
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer_low = optim.Adam(net_low_gamma.parameters(), lr=LEARNING_RATE)
    optimizer_high = optim.Adam(net_high_gamma.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    similarities=[]
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)
       # epsilon=0.01
    
        reward = agent.play_step(
            net_low_gamma=net_low_gamma,
            net_high_gamma=net_high_gamma,
            epsilon=epsilon,
            device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards)
            
            if len(buffer) > 100:
                for _ in range(0,100):
                    optimizer_low.zero_grad(); net_low_gamma.zero_grad()
                    optimizer_high.zero_grad(); net_high_gamma.zero_grad()

                    batch = buffer.sample(BATCH_SIZE)

                    loss_low = calc_loss(batch, net_low_gamma, tgt_net_low_gamma, GAMMA_LOW, device=device)
                    loss_low.backward(retain_graph=False)
                    optimizer_low.step()

                    loss_high = calc_loss(batch, net_high_gamma, tgt_net_high_gamma, GAMMA_HIGH, device=device)
                    loss_high.backward(retain_graph=False)
                    optimizer_high.step()


            if len(total_rewards) % 5 ==0:
                print("%d: done %d games, reward %.3f, "
                      "eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), m_reward, epsilon,
                    speed
                ))


        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            with torch.no_grad():
                del tgt_net_low_gamma, tgt_net_high_gamma #LOAD_STATE_DICT causes memory issues (only seen when running "dmesg" in terminal)
                
                tgt_net_low_gamma = DQN(env.observation_space.shape,
                            env.action_space.n).to('cpu')
                tgt_net_low_gamma.load_state_dict(net_low_gamma.state_dict())

                tgt_net_high_gamma = DQN(env.observation_space.shape,
                            env.action_space.n).to('cpu')
                tgt_net_high_gamma.load_state_dict(net_high_gamma.state_dict())
