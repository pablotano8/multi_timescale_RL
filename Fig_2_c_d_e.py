import pickle
import random as rand

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from scipy.signal import savgol_filter

# PG Network
class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape),
        )

    def forward(self, x):
        out = self.fc(x)
        return out

# Compute rewards-to-go for each timestep to be used in the PG learning target.
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs

# Class definition for the training process and agent behavior. 
class Trainable:
    def __init__(
        self, gammas: list, total_time=12, target=None, possible_targets=None, max_reward_magn = 6
    ) -> None:
        self.total_time = total_time # Length of the linear MDP
        self.gammas = gammas # Discounts
        self.target = target # Method to compute the learning target given the episode's reward time and magnitude
        self.possible_targets = possible_targets # Possible target for the decoder to choose from (e.g. reward times)
        self.max_reward_magn = max_reward_magn # Maximum reward magniture
        self.decoder_net = Decoder(len(gammas), len(possible_targets)) # PG network
        self.optimizer = optim.Adam(self.decoder_net.parameters(), lr=1e-3)  # Optimizer for PG network

    # make function to compute action distribution
    def get_policy(self, obs):
        logits = self.decoder_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    # make loss function whose gradient is policy gradient
    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # Train the PG network
    def train(
        self,
        noise_values=0, # Corrupt values after learning them (not used in the paper)
        epochs=500, # Number of episodes to train for.
        batch_size=100, # Update frequency of the PG network,
        random_reward=False, # Whether the reward magnitude is random
        corr_noise_values=False, # Correlation to corrupt values (not used in the paper)
        noise_perceived_time=False, # Noise in the perceived reward time (not used in the paper)
        noise_reward=0, # Noise in the perceived reward magnitude (not used in the paper)
        min_num_td_steps=59 , # Minimum number of backups to do TD learning in each episode
        max_num_td_steps=99 # Maximum number of backups to do TD learning in each episode
    ):
        total_reward = []

        for ep in range(epochs):
            # make some empty lists for logging.
            batch_obs, batch_acts, batch_weights, corrects = [], [], [], []
            done, ep_rews = False, []

            # iterate until batch is full:
            while True:

                # MDP and Tabular TD learnig parameters
                alpha = np.random.normal(loc=0.1, scale=0.001) # Sample random learning rate in each episode
                td_it = np.random.choice(range(min_num_td_steps, max_num_td_steps))  # Sample random number of backups rate in each episode
                rew_time = np.random.choice(range(1, self.total_time - 1)) # Sample episode's reward time

                # Sample episode's reward magnitude (should always be >0)
                if random_reward:
                    rew_magn = 1+np.random.choice(self.max_reward_magn)
                else:
                    rew_magn = abs(np.random.normal(1, noise_reward))

                # Initialize values of the cue
                values = np.zeros((self.total_time, len(self.gammas)))

                # Tabular TD learning over the MDP
                for _ in range(td_it):

                    if noise_perceived_time:
                        perceived_rew_time = int(
                            abs(
                                rew_time
                                + np.random.normal(0, rew_time * noise_perceived_time)
                            )
                        )
                    else:
                        perceived_rew_time = rew_time

                    for i in reversed(range(self.total_time - 1)):
                        for j in reversed(range(len(self.gammas))):
                            if i == (perceived_rew_time):
                                rew = rew_magn
                            else:
                                rew = 0

                            if i == (self.total_time - 2):
                                values[i, j] = values[i, j] + alpha * (
                                    rew - values[i, j]
                                )
                            else:
                                values[i, j] = values[i, j] + alpha * (
                                    rew
                                    + self.gammas[j] * values[i + 1, j]
                                    - values[i, j]
                                )

                # Corrupt values
                noise = np.random.normal(scale=noise_values * values[0, 0])

                for gamma in range(len(self.gammas)):
                    if corr_noise_values is False:
                        noise = np.random.normal(scale=noise_values)
                    values[0, gamma] = values[0, gamma] + noise

                # The obervation for PG net is values at the cue, it will be used in the PG training batch later
                obs = [list(np.array([values[0, :]]).flatten())]
                obs = np.array([item for sublist in obs for item in sublist])
                obs = obs.flatten()

                # Save obs in the PG training batch
                batch_obs.append(obs)

                # The variable 'plan' is used to compute performance (no exploration)
                plan = self.get_action(
                        torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                    )
                
                # The variable 'act' is the actual decision of the network during training, it has epsilon-greedy for exploration
                # Epsilon-greeduy (eps=0.3)
                if np.random.rand()<0.3:
                    act = np.random.choice(len(self.possible_targets))
                else:
                    act = self.get_action(
                        torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                    )

                # The variable 'correct' is the tracked network performance without exploration (not used to train the network)
                # The variable 'reward' is the reward signal of the true exploratory policy, used to train the network
                rew, correct = 0, 0
                if self.possible_targets[act] == self.target(rew_time, rew_magn):
                    rew = 1

                if self.possible_targets[plan] == self.target(rew_time, rew_magn):
                    correct = 1

                done = True

                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)
                corrects.append(correct)

                if done:
                    # Rewards in the batch
                    batch_weights += list(reward_to_go(ep_rews))

                    # reset episode-specific variables
                    done, ep_rews = False, []

                    # end experience loop if we fill the batch
                    if len(batch_obs) > batch_size:
                        break

                # take a single policy gradient update step
                self.optimizer.zero_grad()
                batch_loss = self.compute_loss(
                    obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(
                        "cpu"
                    ),
                    act=torch.as_tensor(np.array(batch_acts), dtype=torch.int32).to(
                        "cpu"
                    ),
                    weights=torch.as_tensor(
                        np.array(batch_weights), dtype=torch.float32
                    ).to("cpu"),
                )
                batch_loss.backward()
                self.optimizer.step()

            total_reward.append(np.nanmean(corrects))

            if ep % 50 == 0:
                print(f"epoch: {ep} , performance: {np.nanmean(corrects)} ")

        return total_reward


# Compute possible targets for each of the experiments
class Target:
    def __init__(self, discount_type: str, discount_param=0.9) -> None:
        self.discount_type = discount_type
        self.discount_param = discount_param

    def compute_target(self, rew_time, rew_magn):
        if self.discount_type == "hyperbolic":
            return rew_magn * (1 / (1 + self.discount_param * rew_time))
        elif self.discount_type == "delta":
            return rew_time

    def possible_targets(self, max_rew_time, max_rew_magn=6):
        targets = []
        if self.discount_type == "delta":
            for t in range(max_rew_time):
                targets.append(self.compute_target(t, 1))
        else:
            for t in range(max_rew_time):
                for r in range(max_rew_magn):
                    targets.append(self.compute_target(t, 1+r))
        return targets


def plot_performance(perf, gamma_experiments, title="", label=""):

    data = []
    for gamma in gamma_experiments:
        for p in perf[str(gamma)][-100:-1]:
            data.append({'Gamma': str(gamma), 'Performance': p})
    df = pd.DataFrame(data)

    # Create seaborn catplot
    sns.set(style="white")
    g = sns.catplot(x="Gamma", y="Performance", kind="point", errorbar='sd',data=df)
    g.fig.set_size_inches(3, 2) 
    g.fig.suptitle(title)
    # plt.ylim([0.55,1])
    g.set_ylabels("Performance")
    g.ax.tick_params(axis='y', labelsize=16)  # adjust size as needed

    plt.xticks(rotation=45, ha="right")
    return g.fig  # return figure



if __name__ == "__main__":

    # For Fig 2d use 'hyperbolic'. For Fig 2c and 2e use 'delta'
    for discount_type in ["delta"]:

        print(f"------------- START EXPERIMENT {discount_type} -------------")
        gamma_experiments = np.array(
            [[0.6, 0.9, 0.99], [0.6,0.6,0.9], [0.6,0.6,0.99], [0.9,0.9,0.99], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9], [0.99, 0.99, 0.99] ]
        )
        target = Target(discount_type=discount_type)

        performance_experiment = {}
        learning_curves = {} 
        for gammas in gamma_experiments:
            print(gammas)
            experiment = Trainable(
                gammas=gammas,
                total_time=15, # For Fig 2c use 15, else use 8
                max_reward_magn=15, # For Fig 2c use 10, for Fig 2d use 4, for Fig 2e use 1
                target=target.compute_target,
                possible_targets=target.possible_targets(max_rew_time=15,max_rew_magn=15), #Adjust accordingly for each figure
                # (15,10) for Fig 2c; (8,4) for Fig 2d; (8,1) for Fig 2e
            )
            performance_experiment[str(gammas)] = experiment.train(
                noise_values=0,
                epochs=1000,
                random_reward=True, # For Fig 2e use False, else True
                noise_perceived_time=0,
                min_num_td_steps=59, # For incomplete learning (Fig 2e) use 1, else 59
                max_num_td_steps=99
            )
            print(f"------------- END EXPERIMENT {gammas} -------------")

        # Save experimen resultst:
        with open(f"experiment.pkl", "wb") as f:
            pickle.dump(performance_experiment, f)

    # Load experiment:
    with open("experiment.pkl", "rb") as f:
        experiment = pickle.load(f)

    mean_perf = [np.mean(experiment[str(gammas)][-50:-1]) for gammas in gamma_experiments]
    gamma_plots = [
        str(gamma_experiments[4:8][np.argmax(mean_perf[4:8])]),
        str(gamma_experiments[1:4][np.argmax(mean_perf[1:4])]),
        '[0.6  0.9  0.99]']

    fig = plot_performance(experiment, gamma_plots, label="experiment")
    fig.savefig("Fig_2_plot.svg", bbox_inches="tight")

###### Plotting Methods for Ext. Fig 1 ######
def smooth_curve(values, window_size):
    """Computes the moving average and standard error over a window."""
    values_padded = np.pad(values, (window_size // 2, window_size - window_size // 2), mode='edge')
    smoothed_values = np.convolve(values_padded, np.ones(window_size)/window_size, mode='valid')
    return smoothed_values

def compute_error(values, window_size):
    """Computes the standard error over a window."""
    errors = []
    for i in range(window_size // 2, len(values) - window_size // 2):
        window = values[i-window_size//2 : i+window_size//2]
        errors.append(np.std(window) )
    return np.array(errors)



unique_gamma_counts = [len(np.unique(exp)) for exp in gamma_experiments]

colors = []
for unique_gamma_count in unique_gamma_counts:
    if unique_gamma_count == 1:
        colors.append(cm.Blues)
    elif unique_gamma_count == 2:
        colors.append(cm.Greens)
    elif unique_gamma_count == 3:
        colors.append(cm.gray)

window_size = 200
plt.figure(figsize=(5, 4))

# Assume performance_experiment is already defined
for i, ((gamma, curve), color_map) in enumerate(zip(performance_experiment.items(), colors)):
    # Map the range [0, 1] to [0.4, 1] to avoid too light colors
    color = color_map(0.4 + 0.6 * (i / len(gamma_experiments)))
    smoothed_curve = savgol_filter(curve, 200, 1)
    error = compute_error(smoothed_curve, window_size)
    
    min_length = min(len(smoothed_curve), len(error))
    smoothed_curve = smoothed_curve[:min_length]
    error = error[:min_length]
    
    x = np.arange(len(smoothed_curve))
    plt.plot(x, smoothed_curve, label=f"{gamma}", color=color)
    plt.fill_between(x, smoothed_curve - error, smoothed_curve + error, color=color, alpha=0.2)

plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.title('Learning Curves During Training')
# Set the fontsize for the legend
plt.legend(fontsize='x-small')  # you can use 'x-small', 'small', 'medium', etc.
# Save the figure as SVG before showing the plot
plt.savefig('learning_curves.svg', format='svg')
plt.show()