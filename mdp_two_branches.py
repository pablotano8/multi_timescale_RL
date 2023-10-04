import pickle
import random as rand

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from myopic_mdp import generate_mdp,q_learning

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
        self, gammas: list
    ) -> None:
        self.gammas = gammas
        self.decoder_net = Decoder(2*len(gammas), 2)
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
        epochs=500,
        batch_size=100,
        min_num_td_steps=59,
        max_num_td_steps=99
    ):
        total_reward = []

        for ep in range(epochs):
            # make some empty lists for logging.
            batch_obs, batch_acts, batch_weights, corrects = [], [], [], []
            done, ep_rews = False, []

            # iterate until batch is full:
            while True:

                # MDP and Tabular TD learnig parameters
                alpha = np.random.normal(loc=0.1, scale=0.001)
                td_it_up = np.random.choice(range(min_num_td_steps, max_num_td_steps))
                td_it_down = np.random.choice(range(min_num_td_steps, max_num_td_steps))

                while True:
                    branch_length_up = rand.randint(5, 15)
                    branch_length_down = rand.randint(5, 15)
                    if branch_length_up != branch_length_down:
                        break
                    
                # Initialize values
                values_up = np.zeros((15, len(self.gammas)))
                values_down = np.zeros((15, len(self.gammas)))

                # Tabular TD learning over the MDP
                for _ in range(td_it_up):

                    for i in reversed(range(branch_length_up - 1)):
                        for j in reversed(range(len(self.gammas))):

                            if i == (branch_length_up - 2):
                                values_up[i, j] = values_up[i, j] + alpha * (
                                    1 - values_up[i, j]
                                )
                            else:
                                values_up[i, j] = values_up[i, j] + alpha * (
                                    0
                                    + self.gammas[j] * values_up[i + 1, j]
                                    - values_up[i, j]
                                )

                # Tabular TD learning over the MDP
                for _ in range(td_it_down):

                    for i in reversed(range(branch_length_down - 1)):
                        for j in reversed(range(len(self.gammas))):

                            if i == (branch_length_down - 2):
                                values_down[i, j] = values_down[i, j] + alpha * (
                                    1 - values_down[i, j]
                                )
                            else:
                                values_down[i, j] = values_down[i, j] + alpha * (
                                    0
                                    + self.gammas[j] * values_down[i + 1, j]
                                    - values_down[i, j]
                                )


                # obervation for PG net is values at first state
                obs = [list(np.array(np.concatenate([values_up[0, :],values_down[0, :]])).flatten())]
                obs = np.array([item for sublist in obs for item in sublist])
                obs = obs.flatten()

                # save obs
                batch_obs.append(obs)

                # act = np.random.choice(range(self.total_time - 1))
                plan = self.get_action(
                        torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                    )
                if np.random.rand()<0.3:
                    act = np.random.choice(2)
                else:
                    act = self.get_action(
                        torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                    )

                rew, correct = 0, 0
                if act==0 and branch_length_down<branch_length_up:
                    rew=1
                elif act==1 and branch_length_down>branch_length_up:
                    rew=1

                if plan==0 and branch_length_down<branch_length_up:
                    correct=1
                elif plan==1 and branch_length_down>branch_length_up:
                    correct=1

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


import pandas as pd
import seaborn as sns

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
            [[0.6, 0.9, 0.99], [0.6,0.6,0.9], [0.6,0.6,0.99], [0.9,0.9,0.99], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9], [0.99, 0.99, 0.99] ]
        )

        performance_experiment = {}
        learning_curves = {} 
        for gammas in gamma_experiments:
            print(gammas)
            experiment = Trainable(
                gammas=gammas,
            )
            performance_experiment[str(gammas)] = experiment.train(
                epochs=2000,
                min_num_td_steps=1,
                max_num_td_steps=99
            )
            print(f"------------- END EXPERIMENT {gammas} -------------")

        # # Save experiment:
        with open(f"experiment.pkl", "wb") as f:
            pickle.dump(performance_experiment, f)

    # Load experiment:
    with open("experiment.pkl", "rb") as f:
        experiment = pickle.load(f)
    # with open("delta.pkl", "rb") as f:
    #     delta = pickle.load(f)
    # with open("step.pkl", "rb") as f:
    #     step = pickle.load(f)

    mean_perf = [np.mean(experiment[str(gammas)][-50:-1]) for gammas in gamma_experiments]
    gamma_plots = [
        str(gamma_experiments[4:8][np.argmax(mean_perf[4:8])]),
        str(gamma_experiments[1:4][np.argmax(mean_perf[1:4])]),
        '[0.6  0.9  0.99]']

    fig = plot_performance(experiment, gamma_plots, label="experiment")
    fig.savefig("random_magnitude.svg", bbox_inches="tight")


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

window_size = 300
plt.figure(figsize=(5, 4))

# Assume performance_experiment is already defined
for i, ((gamma, curve), color_map) in enumerate(zip(performance_experiment.items(), colors)):
    # Map the range [0, 1] to [0.4, 1] to avoid too light colors
    color = color_map(0.4 + 0.6 * (i / len(gamma_experiments)))
    smoothed_curve = savgol_filter(curve, 300, 1)
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