import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
from IPython.display import clear_output
import matplotlib.pyplot as plt

class GridWorldEnv(gym.Env):
    def __init__(self):
        self.size = 10
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=self.size - 1, shape=(2,), dtype=np.float32)
        self.step_counter = 0

    def reset(self):

        self.state = np.array(np.random.choice(range(0, self.size), 2), dtype=np.float32)

        self.step_counter = 0
        return self.state

    def step(self, action):
        if action == 0:  # up
            self.state[1] = min(self.state[1] + 1, self.size - 1)
        elif action == 1:  # down
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 2:  # right
            self.state[0] = min(self.state[0] + 1, self.size - 1)
        elif action == 3:  # left
            self.state[0] = max(self.state[0] - 1, 0)

        pos_x, pos_y = self.state

        if pos_x == self.size - 2 and pos_y == self.size - 5:
            r = 4
        elif pos_x == self.size - 2 and pos_y == self.size - 9:
            r = -4
        elif pos_x == self.size - 2 and pos_y == self.size - 1:
            r = -4
        else:
            r = np.random.normal(0, 0.05)

        self.step_counter += 1
        done = self.step_counter >= 25
        return self.state, r, done, {}, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass




print('Generating Trajectories...')
# Generate Trajectories
def move(size,pos_x,pos_y,act_x,act_y):
    return max((min((pos_x+act_x,size-1)),0)), max((min((pos_y+act_y,size-1)),0))

num_episodes=10000
size=10
pos_x , pos_y = 0 , 0
length_trajectory=25

POS, POS_NEXT, R, TER, ACT =[] , [] , [] , [] , []
for _ in range(num_episodes):
    count=0
    positions, positions_next, rewards, termination, actions = [], [], [], [] , []
    while True:
        count+=1
            
        if count<length_trajectory:
            act = np.random.choice([0,1,2,3])

            pos_x_next , pos_y_next = pos_x, pos_y
            if act == 0: # up
                pos_y_next = min(size - 1, pos_y + 1)
            elif act == 1: # down
                pos_y_next = max(0, pos_y - 1)
            elif act == 2: # right
                pos_x_next = min(size - 1, pos_x + 1)
            elif act == 3: # left
                pos_x_next = max(0, pos_x - 1)

            if pos_x == size - 2 and pos_y == size - 5:
                r = 4
            elif pos_x == size - 2 and pos_y == size - 9:
                r = -4
            elif pos_x == size - 2 and pos_y == size - 1:
                r = -4
            else:
                r = np.random.normal(0, 0.05)
                
            positions.append([pos_x,pos_y])
            pos_x , pos_y = pos_x_next , pos_y_next
            positions_next.append([pos_x_next,pos_y_next])
            rewards.append(r)
            actions.append(act)
            termination.append(0)
        else:
            termination.append(1)
            pos_x , pos_y = np.random.choice(range(0,size),2)
            positions.append([pos_x,pos_y])
            actions.append(act)
            rewards.append(r)
            positions_next.append([pos_x,pos_y])
            break
    POS.append(positions)
    POS_NEXT.append(positions_next)
    R.append(rewards)
    ACT.append(actions)
    TER.append(termination)
print('Done')


# Register the environment
gym.envs.register(id='GridWorld-v0', entry_point=GridWorldEnv, max_episode_steps=50)

# Training parameters
size = 10
lr = 5e-4
epochs = 100
change = 50
batch_size = 1000
epsilon = 0.1
GAMMAS = [0.6, 0.9, 0.99]
number_of_trajectories = 3
NUM_GAMMAS = len(GAMMAS)

env = gym.make('GridWorld-v0')
num_actions = 4

# Policy network
class policy_net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(policy_net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# Functions for policy gradient
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

Q_gamma = np.zeros((size, size, NUM_GAMMAS, num_actions))

logits_net = policy_net(1 + num_actions * NUM_GAMMAS, num_actions).to('cpu')

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
    batch_first_positive_rew = []

    # reset episode-specific variables
    obs = env.reset()   # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep
    
    while True:

        pos = env.state
        obs = np.array([pos[0] > (size - 1)/2 ] +
                       list(Q_gamma[int(obs[0]), int(obs[1]), :, :].flatten()))
        
        batch_obs.append(obs)

        # act = get_action(torch.as_tensor(obs, dtype=torch.float32).to('cpu'))
        if pos[1] > 5:
            act = np.argmax(Q_gamma[int(pos[0]), int(pos[1]), 2, :])
        else:
            act = np.argmax(Q_gamma[int(pos[0]), int(pos[1]), 2, :])

        obs, rew, done, _, _ = env.step(act)
        
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            ep_ret, ep_len, first_positive_rew = sum(ep_rews), len(ep_rews), next((i for i, x in enumerate(ep_rews) if x == 4), None)
            if (first_positive_rew == None) or (-4 in ep_rews):
                first_positive_rew = len(ep_rews)
            
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_first_positive_rew.append(first_positive_rew)

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
    # optimizer.zero_grad()
    # batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to('cpu'),
    #                             act=torch.as_tensor(batch_acts, dtype=torch.int32).to('cpu'),
    #                             weights=torch.as_tensor(batch_weights, dtype=torch.float32).to('cpu')
    #                             )
    # batch_loss.backward()
    # optimizer.step()
    return batch_loss, batch_rets, batch_lens, batch_first_positive_rew

total_reward = []

for i in range(epochs):
    clear_output(wait=True)

    Q_gamma = np.zeros((size, size, NUM_GAMMAS, num_actions))


    # Sample N trajectories and learn the Q_values

    which_trajectories = []
    while len(which_trajectories) < 1:
        candidate_trajectory = np.random.choice(range(10000))
        if 4 in R[candidate_trajectory]:
            which_trajectories.append(candidate_trajectory)
    while len(which_trajectories) < number_of_trajectories:
         candidate_trajectory = np.random.choice(range(10000))
         which_trajectories.append(candidate_trajectory)



    for g in range(0,NUM_GAMMAS):
        for _ in range(10000):
            
            which_trajectory=np.random.choice(which_trajectories)
            sample=np.random.choice(range(length_trajectory))
            
            ter = TER[which_trajectory][sample]
            pos_x , pos_y = POS[which_trajectory][sample]
            pos_x_next , pos_y_next = POS_NEXT[which_trajectory][sample]
            r = R[which_trajectory][sample]
            act = ACT[which_trajectory][sample]

            if ter==0:
                Q_gamma[pos_x,pos_y,g,act] = Q_gamma[pos_x,pos_y,g,act] + 0.01 * (r + GAMMAS[g] * np.max(
                                                                            Q_gamma[pos_x_next,pos_y_next,g,:]) - Q_gamma[pos_x,pos_y,g,act])
            else:
                Q_gamma[pos_x,pos_y,g,act] = Q_gamma[pos_x,pos_y,g,act] + 0.01 * (r  - Q_gamma[pos_x,pos_y,g,act])


    def euclidean_distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    visited_states = set()
    for which_trajectory in which_trajectories:
        for sample in range(length_trajectory):
            pos_x, pos_y = POS[which_trajectory][sample]
            visited_states.add((pos_x, pos_y))

    for pos_x in range(Q_gamma.shape[0]):
        for pos_y in range(Q_gamma.shape[1]):
            if (pos_x, pos_y) not in visited_states:
                for g in range(NUM_GAMMAS):
                    total_weight = 0
                    weighted_sum = np.zeros(Q_gamma.shape[-1])
                    for which_trajectory in which_trajectories:
                        for sample in range(length_trajectory):
                            pos_x_sample, pos_y_sample = POS[which_trajectory][sample]
                            weight = 1 / euclidean_distance((pos_x, pos_y), (pos_x_sample, pos_y_sample))
                            total_weight += weight
                            weighted_sum += weight * Q_gamma[pos_x_sample, pos_y_sample, g, :]
                    Q_gamma[pos_x, pos_y, g, :] = weighted_sum / total_weight
                
    batch_loss, batch_rets, batch_lens, batch_first_positive_rew = train_one_epoch()

    print('epoch: %3d \t loss: %.3f \t return: %.3f \t first_pos_rew: %.3f' %
            (i, batch_loss, np.mean(batch_rets), np.mean(batch_first_positive_rew)))

    total_reward.append(np.mean(batch_first_positive_rew))

print(np.mean(total_reward))


