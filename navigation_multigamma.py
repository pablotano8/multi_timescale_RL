from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np
import gym
from gym.spaces import Discrete, Box
from mazelab.generators import random_maze



import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color


class Maze(BaseMaze):
    @property
    def size(self):
        return x.shape
    
    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal
    
from mazelab import BaseEnv
from mazelab import VonNeumannMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete


class Env(BaseEnv):
    def __init__(self):
        super().__init__()
        
        self.maze = Maze()
        self.motions = VonNeumannMotion()
        
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        
    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]
        
        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), reward, done, {}, {}
        
    def reset(self):
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        return self.maze.to_value()
    
    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable
    
    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    
    def get_image(self):
        return self.maze.to_rgb()
    



env_name='RandomMaze-v0'
gym.envs.register(id=env_name, entry_point=Env, max_episode_steps=200)

#######################################################################
size = 15
lr=5e-4
#     lr=1e-2
epochs=20000
change = 100
batch_size=1000
epsilon=0.1
GAMMAS = [0.6,0.9,0.99]
#######################################################################
NUM_GAMMAS = len(GAMMAS)

# make environment, check spaces, get obs / act dims
x = random_maze(width=size, height=size, complexity=1, density=0.5)
num_actions = 4
env = gym.make('RandomMaze-v0')

class policy_net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(policy_net, self).__init__()


        self.fc = nn.Sequential(
            nn.Linear(input_shape, 50),
            nn.ReLU(),
            nn.Linear(50, n_actions)
        )

    def forward(self, x):
        return self.fc(x)


# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
#         logits = obs
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


Q_gamma = np.zeros((size,size,NUM_GAMMAS,num_actions))

logits_net = policy_net(4+num_actions*NUM_GAMMAS,num_actions).to('cpu')

# make optimizer
optimizer = optim.Adam(logits_net.parameters(), lr=lr)

# for training policy
def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep
    
    while True:
        pos= env.maze.objects.agent.positions[0]*1
        r_pos = env.maze.objects.goal.positions[0]*1

        obs = [[(pos[0]-size/2)/25,(pos[1]-size/2)/25,(r_pos[0]-size/2)/25,(r_pos[1]-size/2)/25],
                list(Q_gamma[pos[0],pos[1],:,:].flatten())]
        obs = np.array([item for sublist in obs for item in sublist])
        obs = obs.flatten()


        # save obs
        batch_obs.append(obs)

        # Choose action
        if any([i<5,(i)%change==0,(i-1)%change==0,(i-2)%change==0,(i-3)%change==0,(i-4)%change==0,(i-5)%change==0,(i-6)%change==0,
                (i-7)%change==0,(i-8)%change==0,(i-9)%change==0,(i-10)%change==0 ,(i-11)%change==0]):
            act = np.random.choice([0,1,2,3])
        else:
            act = get_action(torch.as_tensor(obs, dtype=torch.float32).to('cpu'))

        _, rew, done, _, _ = env.step(act)
    
        pos_next = env.maze.objects.agent.positions[0]*1  
            
        utility = pos_next[0]==r_pos[0] and pos_next[1]==r_pos[1]

        for g in range(NUM_GAMMAS):
            Q_gamma[pos[0],pos[1],g,act] = Q_gamma[pos[0],pos[1],g,act] + np.random.normal(0.1,0.01) * (utility + np.random.normal(0,0.01) +  (1 - utility) * GAMMAS[g] * np.max(
                                                                        Q_gamma[pos_next[0],pos_next[1],g,:]) - Q_gamma[pos[0],pos[1],g,act])

        
        if rew==1:
            rew=50
        if rew==-1:
            rew=-0.2
            
        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)


        if done:
            if any([i==2, i==3, i==4, i==10, (i-2)%change==0,(i-3)%change==0,(i-4)%change==0,(i-10)%change==0,(i-11)%change==0]):
                start_idx = [L[np.random.randint(0, len(L))]]
        
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            
            # the weight for each logprob(a|s) is R(tau)
            batch_weights += list(reward_to_go(ep_rews))

            # reset episode-specific variables
#             start_idx = [L[np.random.randint(0, len(L))]]
            obs, done, ep_rews = env.reset(), False, []


            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    batch_loss = 0
    if all([i>5,(i)%change!=0,(i-1)%change!=0 , (i-2)%change!=0 , (i-3)%change!=0 , (i-4)%change!=0 , (
        i-5)%change!=0,(i-6)%change!=0 , (i-7)%change!=0 , (i-8)%change!=0 , (i-9)%change!=0 , (i-10)%change!=0 , (i-11)%change!=0]):
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to('cpu'),
                                    act=torch.as_tensor(batch_acts, dtype=torch.int32).to('cpu'),
                                    weights=torch.as_tensor(batch_weights, dtype=torch.float32).to('cpu')
                                    )
        batch_loss.backward()
        optimizer.step()
    return batch_loss, batch_rets, batch_lens

total_reward = []
# training loop

for i in range(epochs):

    clear_output(wait=True)
    if i==0 or i%change==0:
        Q_gamma = np.zeros((size,size,NUM_GAMMAS,num_actions))

        x = random_maze(width=size, height=size, complexity=0.1, density=0.1)
        env = gym.make(env_name)
        L = env.maze.objects.free.positions
        goal_idx = [L[np.random.randint(0, len(L))]]

        start_idx = goal_idx

    elif i in (1,2,12) or (i-1)%change==0 or (i-2)%change==0 or (i-12)%change==0: 
        start_idx = goal_idx
    else:
        start_idx = [L[np.random.randint(0, len(L))]]

    
    obs = env.reset()
    batch_loss, batch_rets, batch_lens = train_one_epoch()

    if i%10==0:
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

    if i>5 and (i)%change!=0 and (i-1)%change!=0 and (i-2)%change!=0 and (i-3)%change!=0 and (i-4)%change!=0 and (
        i-5)%change!=0 and (i-6)%change!=0 and (i-7)%change!=0 and (i-8)%change!=0 and (i-9)%change!=0 and (i-10)%change!=0 and (i-11)%change!=0 and (i-12)%change!=0:
        total_reward.append(np.mean(batch_lens))

with open('nav_06_no_detection.npy', 'wb') as f:
    np.save(f, total_reward)



# import torch.autograd as autograd

# # Saliency computation function
# def compute_saliency(position):
#     obs = env.maze.to_value()
#     env.maze.objects.agent.positions = [position]
#     pos = env.maze.objects.agent.positions[0] * 1
#     r_pos = env.maze.objects.goal.positions[0] * 1

#     obs = [[(pos[0] - size / 2) / 25, (pos[1] - size / 2) / 25, (r_pos[0] - size / 2) / 25, (r_pos[1] - size / 2) / 25],
#            list(Q_gamma[pos[0], pos[1], :, :].flatten())]
#     obs = np.array([item for sublist in obs for item in sublist])
#     obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
#     obs_tensor.requires_grad = True

#     logits = logits_net(obs_tensor)
#     action_probs = get_policy(obs_tensor)
#     most_probable_action = action_probs.probs.argmax().item()

#     logits[most_probable_action].backward()
#     saliency_map = obs_tensor.grad.data.abs().numpy()
#     return saliency_map

# # Function to plot the most important Q-value input's gamma
# def plot_important_gamma(position, saliency_map):
#     q_values_start_idx = 4
#     important_q_value_idx = np.argmax(saliency_map[q_values_start_idx:]) + q_values_start_idx
#     gamma_idx = (important_q_value_idx - q_values_start_idx) // num_actions
#     important_gamma = GAMMAS[gamma_idx]

#     plt.scatter(position[1], position[0], c=important_gamma, cmap='viridis', marker='s', s=200)
#     plt.colorbar()

# plt.figure(figsize=(10, 10))
# plt.imshow(env.maze.to_rgb())
# free_positions = env.maze.objects.free.positions

# for position in free_positions:
#     saliency_map = compute_saliency(position)
#     plot_important_gamma(position, saliency_map)

# plt.title("Most Important Gamma per Position")
# plt.gca().invert_yaxis()
# plt.show()
