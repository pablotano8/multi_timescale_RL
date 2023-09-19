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
        random_reward=False,
        corr_noise_values=False,
        noise_perceived_time=False,
        noise_reward=0,
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
                td_it = np.random.choice(range(1, 99))
                rew_time = np.random.choice(range(1, self.total_time - 1))

                if random_reward:
                    rew_magn = 1+np.random.choice(self.max_reward_magn)
                else:
                    rew_magn = abs(np.random.normal(1, noise_reward))

                # Initialize values
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

                            if i == (self.total_time - 1):
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

                # act = np.random.choice(range(self.total_time - 1))
                act = np.random.choice(len(self.possible_targets))

                plan = self.get_action(
                    torch.as_tensor(obs, dtype=torch.float32).to("cpu")
                )

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

    def compute_target(self, rew_time, rew_magn):
        if self.discount_type == "hyperbolic":
            return rew_magn * (1 / (1 + self.discount_param * rew_time))
        elif self.discount_type == "delta":
            return rew_time
        elif self.discount_type == "exponential":
            return rew_magn * (self.discount_param**rew_time)
        elif self.discount_type == "step":
            if rew_time < 5:
                return rew_magn
            else:
                return 0

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

    # gamma_experiments = np.array(
    #     [[0.6, 0.6, 0.9]]
    # )

    for discount_type in ["delta"]:

        print(f"------------- START EXPERIMENT {discount_type} -------------")
        gamma_experiments = np.array(
            [[0.6, 0.9, 0.99], [0.6,0.6,0.9], [0.6,0.6,0.99], [0.9,0.9,0.99], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9], [0.99, 0.99, 0.99] ]
        )
        target = Target(discount_type=discount_type)

        performance_experiment = {}
        for gammas in gamma_experiments:
            print(gammas)
            experiment = Trainable(
                gammas=gammas,
                total_time=7,
                max_reward_magn=1,
                target=target.compute_target,
                possible_targets=target.possible_targets(max_rew_time=7,max_rew_magn=1),
            )
            performance_experiment[str(gammas)] = experiment.train(
                noise_values=0,
                epochs=1000,
                random_reward=False,
                noise_perceived_time=0.05,
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

# plt.figure(figsize=(4, 3))
# t = np.linspace(0,10,1000)
# plt.plot(t,1 / (1 + 0.9 * t),label="hyperbolic")
# plt.plot(t,t==t[500], label="delta")
# plt.plot(t,0.25*(t>t[250]), label="step")
# plt.legend()
# plt.ylabel('Value')
# plt.xlabel('Time')
# plt.title('Discounts')
# plt.savefig("test.png", bbox_inches="tight")
# plt.show()
