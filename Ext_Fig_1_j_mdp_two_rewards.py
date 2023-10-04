import pickle
import random as rand

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from Fig_2_f_myopic_mdp import generate_mdp,q_learning
from itertools import combinations_with_replacement
import pandas as pd
import seaborn as sns
from itertools import product


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


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


class Trainable:
    def __init__(
        self, gammas: list, total_time=12, target=None, possible_targets=None, max_reward_magn = 6
    ) -> None:
        self.total_time = total_time
        self.gammas = gammas
        self.target = target
        self.possible_targets = possible_targets
        self.max_reward_magn = max_reward_magn
        self.decoder_net = Decoder(len(gammas), len(possible_targets))
        self.optimizer = optim.Adam(self.decoder_net.parameters(), lr=1e-3)

    # make function to compute action distribution
    def get_policy(self, obs):
        logits = self.decoder_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self, obs, act, weights):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def train(
        self,
        noise_values=0,
        epochs=500,
        batch_size=100,
        corr_noise_values=False,
        noise_perceived_time=False,
        min_num_td_steps=59,
        max_num_td_steps=99
    ):
        total_reward = []
        epoch_performance = []

        for ep in range(epochs):
            # make some empty lists for logging.
            batch_obs, batch_acts, batch_weights, corrects = [], [], [], []
            done, ep_rews = False, []

            # iterate until batch is full:
            while True:

                # MDP and Tabular TD learnig parameters
                alpha = np.random.normal(loc=0.1, scale=0.001)
                td_it = np.random.choice(range(min_num_td_steps, max_num_td_steps))
                rew_time1 = np.random.choice(range(1, self.total_time -1 ))
                rew_time2 = np.random.choice(range(1, self.total_time -1 ))

                rew_magn1 = 1 + np.random.choice(self.max_reward_magn)
                rew_magn2 = 1 + np.random.choice(self.max_reward_magn)

                # Initialize values
                values = np.zeros((self.total_time, len(self.gammas)))

                # Tabular TD learning over the MDP
                for _ in range(td_it):

                    for i in reversed(range(self.total_time -1)):
                        for j in reversed(range(len(self.gammas))):
                            if i == rew_time1:
                                rew = rew_magn1
                            elif i== rew_time2:
                                rew = rew_magn2
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

                # obervation for PG net is values at first state
                obs = [list(np.array([values[0, :]]).flatten())]
                obs = np.array([item for sublist in obs for item in sublist])
                obs = obs.flatten()

                # save obs
                batch_obs.append(obs)

                plan = self.get_action(
                        torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                    )
                if np.random.rand()<0.3:
                    act = np.random.choice(len(self.possible_targets))
                else:
                    act = self.get_action(
                        torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                    )
                
                rew, correct = 0, 0
                if self.possible_targets[act] == rew_time1 * rew_time2:
                    rew = 1

                if self.possible_targets[plan] == rew_time1 * rew_time2:
                    correct = 1

                done = True

                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)
                corrects.append(correct)

                if done:

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += list(reward_to_go(ep_rews))

                    # reset episode-specific variables
                    done, ep_rews = False, []

                    # end experience loop if we have enough of it
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


class Target:
    def __init__(self, discount_type: str, discount_param=0.9) -> None:
        self.discount_type = discount_type
        self.discount_param = discount_param

    def compute_target(self, rew_time):
        return rew_time

    def possible_targets(self, max_rew_time):
        return list(set(a * b for a, b in product(range(1, max_rew_time), repeat=2)))




def plot_performance(perf, gamma_experiments, title="", label=""):
    # Prepare data for seaborn
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

    for discount_type in ["delta"]:

        print(f"------------- START EXPERIMENT {discount_type} -------------")
        gamma_experiments = np.array(
            [[0.6,0.7,0.9,0.95,0.99],[0.6, 0.9, 0.99], [0.6,0.6,0.9], [0.6,0.6,0.99], [0.9,0.9,0.99], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9], [0.99, 0.99, 0.99] ]
        )
        target = Target(discount_type=discount_type)

        performance_experiment = {}
        learning_curves = {} 
        for gammas in gamma_experiments:
            print(gammas)
            experiment = Trainable(
                gammas=gammas,
                total_time=5,
                max_reward_magn=3,
                target=target.compute_target,
                possible_targets=target.possible_targets(max_rew_time=5),
            )
            performance_experiment[str(gammas)] = experiment.train(
                noise_values=0,
                epochs=2000,
                noise_perceived_time=0,
                min_num_td_steps=59,
                max_num_td_steps=99
            )
            print(f"------------- END EXPERIMENT {gammas} -------------")

        # # Save experiment:
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

from scipy.signal import savgol_filter

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


from scipy.signal import savgol_filter
import matplotlib.cm as cm

unique_gamma_counts = [len(np.unique(exp)) for exp in gamma_experiments]

colors = []
for unique_gamma_count in unique_gamma_counts:
    if unique_gamma_count == 1:
        colors.append(cm.Blues)
    elif unique_gamma_count == 2:
        colors.append(cm.Greens)
    elif unique_gamma_count == 3:
        colors.append(cm.gray)
    elif unique_gamma_count == 5:
        colors.append(cm.Reds)

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