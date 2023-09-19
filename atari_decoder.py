# import sys  
import argparse
import collections
import os
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score

DEVICE = 'cuda'

DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"

WHICH_LAYER = 'output' #input #conv #FC #output

TARGET_LENGTH=200
ITERATIONS = 25_000
NUM_TRAINING_SAMPLES = 10_000
TEST_SIZE = 1_000

BATCH_SIZE = 32
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info
    
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

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

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

class Classifier(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Classifier, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )

      #  self.fc = nn.Sequential(
      #      nn.Linear(input_shape, output_shape),
      #  )

    def forward(self, x):
        return self.fc(x)

def decode_temporal_evolution(
    net,
    net_sim,
    plot_test=False):

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args,unknown = parser.parse_known_args()
    device = "mps"

    env = BasicWrapper(gym.make_env(args.env))

    buffer = ExperienceBuffer(NUM_TRAINING_SAMPLES)

    print('Preparing training pipeline...')
    buffer.fill_buffer(net = net_sim, env = env, size = NUM_TRAINING_SAMPLES)
    
    print('Buffer filled')
    states=np.array([buffer.buffer[start].state for start in range(0,len(buffer)-TARGET_LENGTH)])
    rewards=np.array([np.array([buffer.buffer[t].reward for t in np.arange(start,start+TARGET_LENGTH)]) for start in range(0,len(buffer)-TARGET_LENGTH)])
    print('States and rewards collected')
    running_loss = 0.0

    print('Ready to train classifier..')


    criterion = nn.MSELoss()
        
    
    after_conv = nn.Sequential(*list(net.children())[0:1]).to('cuda')

    # new_index=[index for index in range(1,len(rewards)-1) if np.prod(rewards[index,:]==0)==0] #indeces without all zero rewards
    new_index = range(1,len(rewards)-1)

    if WHICH_LAYER == 'input':
        input_size = 28224
    elif WHICH_LAYER == 'conv':
        input_size = 3136
    elif WHICH_LAYER == 'FC':
        input_size = 512 * NUM_GAMMAS
    elif WHICH_LAYER == 'output':
        input_size = 4 * NUM_GAMMAS

    classifier = Classifier(input_size,TARGET_LENGTH).to(device) #frames:28224   conv:3136   output:num_actiosns*num_h*num_gammas
    optimizer = optim.SGD(classifier.parameters(), lr=0.001,momentum=0.9)

    train_index=np.random.choice(new_index, len(new_index)-TEST_SIZE,replace=False)
    test_index=list(set(new_index).symmetric_difference(set(train_index)))

    #train_index=np.arange(len(rewards)-TEST_SIZE)
    #test_index=np.arange(len(rewards)-TEST_SIZE,len(rewards))

    ############################# TRAIN #######################################
    for i in range(0,ITERATIONS):
        start=np.random.choice(train_index, BATCH_SIZE)
        indices = np.array([np.arange(start,start+TARGET_LENGTH) for start in start])

        frame = torch.tensor(np.array(
            states[start,:,:,:], copy=False)).to(device)
        
        
        with torch.no_grad():

            if WHICH_LAYER == 'input':
                pre_input_classifier = frame.to('cpu').numpy()
            elif WHICH_LAYER == 'conv':
                pre_input_classifier = after_conv(frame).to('cpu').numpy()
            elif WHICH_LAYER == 'FC':
                conv_out = net.conv(frame).view(frame.size()[0], -1)
                pre_input_classifier = torch.cat([net.layers[i][0:2](conv_out) for i in range(NUM_GAMMAS)],1).detach().to('cpu').numpy()
            elif WHICH_LAYER == 'output': 
                pre_input_classifier = net(frame).detach().to('cpu').numpy()
            
            input_classifier=torch.tensor([pre_input_classifier[i,:].flatten() for i in range(0,BATCH_SIZE)]).to(device)
            target_classifier=torch.tensor(rewards[start,:]).to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(input_classifier).to(device)
        loss = criterion(outputs, target_classifier.float())
        loss.backward()
        optimizer.step()
        

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('iteration: %5d,  training loss: %.3f' %
                (i + 1, running_loss / 1000))
            running_loss = 0.0

    print('Finished Training')


    ############################# TEST #######################################
    print('Starging Testing...')

    running_loss = 0.0
    losses=[]

    for i in range(0,20):
        print(i)
        start=np.random.choice(test_index,999)

        frame = torch.tensor(np.array(
            states[start,:,:,:], copy=False)).to(device)
        
    #   
        with torch.no_grad():
            if WHICH_LAYER == 'input':
                pre_input_classifier = frame.to('cpu').numpy()
            elif WHICH_LAYER == 'conv':
                pre_input_classifier = after_conv(frame).to('cpu').numpy()
            elif WHICH_LAYER == 'FC':
                conv_out = net.conv(frame).view(frame.size()[0], -1)
                pre_input_classifier = torch.cat([net.layers[i][0:2](conv_out) for i in range(NUM_GAMMAS)],1).detach().to('cpu').numpy()
            elif WHICH_LAYER == 'output': 
                pre_input_classifier = net(frame).detach().to('cpu').numpy()
            
            input_classifier=torch.tensor([pre_input_classifier[i,:].flatten() for i in range(0,999)]).to(device)
            target_classifier=torch.tensor(rewards[start,:]).to(device)

            outputs = classifier(input_classifier).to(device)

        loss=r2_score(savgol_filter(np.array(target_classifier.to('cpu').detach()),5,1),
                        savgol_filter(np.array(outputs.to('cpu').detach()),5,1))

        losses.append(loss.item())
    print('R^2 score: ',np.mean(losses))

    if plot_test:
        plt.figure(); 
        plt.plot(savgol_filter(np.array(target_classifier.to('cpu').detach()),5,1),
                        savgol_filter(np.array(outputs.to('cpu').detach()),5,1),'.')
        plt.title(r2_score(np.array(target_classifier.to('cpu').detach()),np.array(outputs.to('cpu').detach())))
        plt.show()

    return np.mean(losses)



if __name__ == "__main__":

    all_decoding_paths = ['dqn_oh_50g_breakout.pt']
    # all_decoding_paths = ['dqn_breakout.pt','dqn_breakout_1.pt','dqn_breakout_2.pt','dqn_breakout_3.pt','dqn_breakout_4.pt']
    sim_path = 'dqn_oh_50g_breakout.pt'

    decoding_performance = []
    for net_path in all_decoding_paths:
        print(net_path)
        NUM_GAMMAS = 50
        net = torch.load(net_path)
        NUM_GAMMAS = 50
        net_sim = torch.load(sim_path)
        NUM_GAMMAS = 50

        decoding_performance.append(decode_temporal_evolution(net,net_sim)) 

    print(('MEAN PERFORMANCE: ',np.mean(decoding_performance)))





    