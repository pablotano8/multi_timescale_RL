# import sys
import collections

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"

DEFAULT_ENV_NAME = "LunarLander-v2"

TARGET_LENGTH = 1000 #Future transitions used to compute empirical Q-values (1000 is the maximum episode length allowed by the game)

NUM_TRAINING_SAMPLES = 25_000 # States sampled to estimate Q_values

# Array of discounts
NUM_GAMMAS=25
GAMMAS = np.linspace(0.6,0.99,NUM_GAMMAS)
GAMMAS=np.flip(GAMMAS)

# Experience class to store transitions for replay buffer
Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

# Function to estimate the Kendall's tau error by sampling num_pairs pairs between esitmates and targets
def kendall_tau_error(estimated_q_values, empirical_q_values, num_pairs=5000):
    """Compute Kendall's tau error."""
    
    num_errors = 0
    indices = list(range(len(estimated_q_values)))
    
    for _ in range(num_pairs):
        # Sample two distinct indices
        i, j = np.random.choice(indices, 2, replace=False)
        
        empirical_order = np.sign(empirical_q_values[i] - empirical_q_values[j])
        estimated_order = np.sign(estimated_q_values[i] - estimated_q_values[j])
        
        if empirical_order != estimated_order:
            num_errors += 1
            
    return num_errors / num_pairs


# Function to compute the empirical discounted reward for a sequence
def compute_discounted_reward(reward_sequence, gamma, done_sequence):
    """Compute discounted reward for a sequence."""
    total_reward = 0
    for i, (reward, done) in enumerate(zip(reward_sequence, done_sequence)):
        total_reward += reward * (gamma ** i)
        if done:
            break
    return total_reward


# Function to compute Q-value accuracy for sequences of states and rewards
def compute_q_value_accuracy_for_sequence(net, states, reward_sequences, actions, dones, gamma_idx):
    """Compute Q-value accuracy for sequences of states and rewards."""
    
    q_values_multi_gamma = net(torch.tensor(states)).detach().cpu().numpy()
    
    # Lists to store all predicted and empirical Q-values
    all_predicted_q_values = []
    all_empirical_q_values = []
    
    # Compute discounted rewards and predicted Q-values for all states
    for i in range(len(states)):
        # Compute discounted reward for the sequence using the specific gamma from GAMMAS
        discounted_reward = compute_discounted_reward(reward_sequences[i], 0.99, dones[i])
        all_empirical_q_values.append(discounted_reward)
        
        # Extract predicted Q-value for the behavioral action for the specific gamma
        action_offset = gamma_idx * 4
        predicted_q_value = q_values_multi_gamma[i][actions[i] + action_offset]
        all_predicted_q_values.append(predicted_q_value)
    
    # Compute Kendall tau-inspired error for all states
    error = kendall_tau_error(all_predicted_q_values, all_empirical_q_values)

    return error

# Function to compute average Q-value accuracy for multiple states, close and far from landing site
def compute_accuracy_for_states(net, net_sim):

    env = gym.make(DEFAULT_ENV_NAME)

    buffer = ExperienceBuffer(NUM_TRAINING_SAMPLES)

    print("Filling buffer...")
    buffer.fill_buffer(net=net_sim, env=env, size=NUM_TRAINING_SAMPLES)

    print("Buffer filled")
    states = np.array(
        [buffer.buffer[start].state for start in range(0, len(buffer) - TARGET_LENGTH - 1)]
    )
    rewards = np.array(
        [
            np.array(
                [
                    buffer.buffer[t].reward
                    for t in np.arange(start, start + TARGET_LENGTH)
                ]
            )
            for start in range(0, len(buffer) - TARGET_LENGTH-1)
        ]
    )

    # Getting the done sequences from the buffer
    dones = np.array(
        [
            np.array(
                [
                    buffer.buffer[t].done
                    for t in np.arange(start, start + TARGET_LENGTH)
                ]
            )
            for start in range(0, len(buffer) - TARGET_LENGTH-1)
        ]
    )


    actions = np.array(
        [buffer.buffer[start].action for start in range(0, len(buffer) - TARGET_LENGTH-1)]
    )


    # Compute the median of the second element across all states (rocket's height)
    medians = np.median(states[:, 1])

    # Split the states, actions, and rewards into two groups according to height
    below_median_indices = np.where(states[:, 1] < medians)[0]
    above_median_indices = np.where(states[:, 1] >= medians)[0]

    states_below_median = states[below_median_indices]
    states_above_median = states[above_median_indices]

    rewards_below_median = rewards[below_median_indices]
    rewards_above_median = rewards[above_median_indices]

    dones_below_median = dones[below_median_indices]
    dones_above_median = dones[above_median_indices]
    actions_below_median = actions[below_median_indices]
    actions_above_median = actions[above_median_indices]

    accuracies_below_median = []
    accuracies_above_median = []

    for gamma_idx in range(NUM_GAMMAS):
        accuracy_below = compute_q_value_accuracy_for_sequence(net, states_below_median, rewards_below_median, actions_below_median,dones_below_median, gamma_idx)
        accuracy_above = compute_q_value_accuracy_for_sequence(net, states_above_median, rewards_above_median, actions_above_median,dones_above_median, gamma_idx)
    
        accuracies_below_median.append(accuracy_below)
        accuracies_above_median.append(accuracy_above)

    return accuracies_below_median, accuracies_above_median

# Define DQN (Deep Q-Network) class
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
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
            nn.Linear(512, NUM_GAMMAS * action_dim),
        )

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
        # return self.fc(x)

# Buffer to store and sample experiences
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )

    def fill_buffer(self, net, env, size):
        state = env.reset()

        while len(self.buffer) < size:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to("cpu")
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v.narrow(-1, 0, env.action_space.n), dim=1)
            action = int(act_v.item())

            # do step in the environment
            new_state, reward, is_done, _ = env.step(action)

            exp = Experience(state, action, reward, is_done, new_state)
            self.buffer.append(exp)
            state = new_state
            if is_done:
                state = env.reset()


# Main function to run the simulation and evaluate Q-value accuracy
if __name__ == "__main__":

    all_error_close, all_error_far = [], []
    unnorm_all_error_close, unnorm_all_error_far = [], []
    NUM_NETS=10
    all_nets = []
    # Iterating over different neural networks. NETWORKS MUST BE TRAINED USING Fig_2_g_train_lunar_multi_gamma.py !!
    for i in range(NUM_NETS):
        print(i)

        # Networks trained using the script Fig_2_g_train_lunar_multi_gamma.py
        net = torch.load( f"nets/lunar_multi_gamma_{i}.pt")
        net_sim = torch.load( f"nets/lunar_multi_gamma_{i}.pt")

        error_close, error_far = compute_accuracy_for_states(net,net_sim)

        normalized_error_close = np.array(error_close) / (np.array(error_close) + np.array(error_far))
        normalized_error_far = np.array(error_far) / (np.array(error_close) + np.array(error_far))

        all_error_close.append(normalized_error_close)
        all_error_far.append(normalized_error_far)

        unnorm_all_error_close.append(np.array(error_close))
        unnorm_all_error_far.append(np.array(error_far))


    mean_close = 1-np.mean(unnorm_all_error_close, axis=0)
    mean_far = 1-np.mean(unnorm_all_error_far, axis=0)

    sem_close = np.std(unnorm_all_error_close, axis=0) / np.sqrt(NUM_NETS)
    sem_far = np.std(unnorm_all_error_far, axis=0) / np.sqrt(NUM_NETS)

    fig, ax = plt.subplots(figsize=(3,3))

    # Plot your data
    matlab_blue = (0, 0.2, 0.6)
    matlab_red = (0.7, 0.1, 0)
    ax.plot(GAMMAS, mean_close, color=matlab_blue)
    ax.plot(GAMMAS, mean_far, color=matlab_red)
    ax.plot(GAMMAS, mean_close, '.', color=matlab_blue)
    ax.plot(GAMMAS, mean_far, '.', color=matlab_red)
    ax.fill_between(GAMMAS, mean_close - sem_close, mean_close + sem_close, color=matlab_blue, alpha=0.2)
    ax.fill_between(GAMMAS, mean_far - sem_far, mean_far + sem_far, color=matlab_red, alpha=0.2)

    # ax.set_ylim(-0.6,0.24)
    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set the remaining spines (left and bottom) to be more pronounced
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Ensure the ticks only appear on the left and bottom
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()

    # Save the figure as an SVG
    fig.savefig('my_figure.svg', format='svg')

