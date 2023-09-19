import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_mdp():
    mdp = {}

    # Initial state
    mdp['s1'] = {'up': ('s2', random.choice([-0.5, 0.5])), 'down': ('b1', random.choice([-0.5, 0.5]))}

    # First upper branch
    for i in range(2, 8):
        mdp[f's{i}'] = {'forward': (f's{i+1}', random.choice([-0.5, 0.5]))}
    mdp['s8'] = {'forward': ('s9', random.choice([-0.5, 0.5]))}
    mdp['s9'] = {'up': ('s10', 1), 'down': ('l1', random.choice([-0.5, 0.5]))}

    # Second upper branch
    for i in range(10, 16):
        mdp[f's{i}'] = {'forward': (f's{i+1}', random.choice([-0.5, 0.5]))}
    mdp['s10'] = {'forward': ('s11', 1)}
    mdp['s15'] = {'forward': (None, random.choice([-0.5, 0.5]))}

    # Lower branch
    for i in range(1, 8):
        mdp[f'b{i}'] = {'forward': (f'b{i+1}', random.choice([-0.5, 0.5]))}
    mdp['b8'] = {'forward': (None, random.choice([-0.5, 0.5]))}

    # Lower branch of the bifurcation
    for i in range(1, 8):
        mdp[f'l{i}'] = {'forward': (f'l{i+1}', random.choice([-0.5, 0.5]))}
    mdp['l8'] = {'forward': (None, random.choice([-0.5, 0.5]))}

    return mdp


def simulate_trajectory(mdp, initial_state, action_s1, action_s9):
    current_state = initial_state
    trajectory = [current_state]
    total_reward = 0

    while current_state is not None:
        if current_state == 's1':
            action = action_s1
        elif current_state == 's9':
            action = action_s9
        else:
            action = 'forward'

        next_state, reward = mdp[current_state][action]
        trajectory.append(next_state)
        total_reward += reward
        current_state = next_state

    return total_reward, trajectory[:-1]

import random

def perform_trajectories(mdp):
    trajectories = [
        ('down', 'down'),
        ('up', 'down'),
        ('up', 'up')
    ]

    memory = []

    for action_s1, action_s9 in trajectories:
        total_reward, trajectory = simulate_trajectory(mdp, 's1', action_s1, action_s9)
        memory.append((trajectory, total_reward))

    return memory

def q_learning(mdp, memory, alpha=0.1, gamma=0.99, episodes=10000):
    q_values = {state: {action: 0 for action in actions} for state, actions in mdp.items()}

    for _ in range(episodes):
        trajectory, total_reward = random.choice(memory)
        for i, state in enumerate(trajectory[:-1]):
            action = 'forward'
            if state == 's1':
                action = 'up' if trajectory[i+1][0] == 's' else 'down'
            elif state == 's9':
                action = 'up' if trajectory[i+1][0] == 's' else 'down'

            next_state = trajectory[i + 1]
            reward = mdp[state][action][1]

            q_current = q_values[state][action]
            q_next_max = max(q_values[next_state].values()) if next_state is not None else 0

            q_values[state][action] += alpha * (reward + gamma * q_next_max - q_current)

    return q_values


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

def compute_accuracy(model, x, y):
    """
    Compute accuracy and SEM for given input, output, and model.
    """
    with torch.no_grad():
        outputs = model(x)
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == y).float().numpy()
        sem = correct.std() / np.sqrt(len(correct))
        return correct.mean(), sem

def train_decoder(training_data, epochs=100, learning_rate=0.001, val_ratio=0.2, batch_size=32, gammas=[0.9]):
    """
    Train the decoder model on given training data and evaluate on a validation set.

    Parameters:
    - training_data (list): List of tuples (input_data, labels).
    - epochs (int): Number of epochs for training.
    - learning_rate (float): Learning rate for the optimizer.
    - val_ratio (float): Fraction of data to be used for validation.
    - batch_size (int): Size of mini-batches for training.

    Returns:
    - model: Trained Decoder model.
    """

    # Convert to PyTorch tensors and split into training and validation sets
    x = torch.FloatTensor([x for x, _ in training_data])
    y = torch.LongTensor([y for _, y in training_data])  # Changed from stack to LongTensor

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio)

    # Define the model, loss, and optimizer
    model = Decoder(input_shape=2 * len(gammas), output_shape=2) # Only 2 q-values per state for each gamma
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Number of batches
    num_batches = len(x_train) // batch_size

    # Training the decoder using mini-batches
    for epoch in range(epochs):
        # Shuffle training data for each epoch
        indices = torch.randperm(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set after each epoch
        val_accuracy_mean, val_accuracy_sem = compute_accuracy(model, x_val, y_val)
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Val Acc: {val_accuracy_mean * 100:.2f}% ± {val_accuracy_sem * 100:.2f}%")

    return model


def generate_multi_gamma_qvalues(mdp, gammas,episodes=1000):
    multi_gamma_q_values = []
    
    for gamma in gammas:
        memory = perform_trajectories(mdp)
        q_values = q_learning(mdp, memory, gamma=gamma,episodes=episodes)
        multi_gamma_q_values.append(q_values)

    return multi_gamma_q_values

def prepare_training_data_for_state(multi_gamma_q_values, state):
    combined_q_values = []

    # Shuffle order to be used for all gammas
    state_values_first_gamma = [(multi_gamma_q_values[0][state]['up'], 0), (multi_gamma_q_values[0][state]['down'], 1)]
    shuffled_state_values = custom_shuffle(state_values_first_gamma)

    for q_values in multi_gamma_q_values:
        state_values = [(q_values[state]['up'], 0), (q_values[state]['down'], 1)]
        
        # Use the same shuffle order for all gammas
        shuffled_state_values_for_current_gamma = [state_values[i] for i in [x[1] for x in shuffled_state_values]]
        
        input_data = [
            shuffled_state_values_for_current_gamma[0][0], shuffled_state_values_for_current_gamma[1][0]
        ]
        
        combined_q_values.extend(input_data)
    
    label = state_values_first_gamma[0][1] if shuffled_state_values[0][0] == multi_gamma_q_values[0][state]['up'] else state_values_first_gamma[1][1]

    return combined_q_values, label


def custom_shuffle(lst):
    idxs = list(range(len(lst)))
    random.shuffle(idxs)
    return [lst[i] for i in idxs]



if __name__ == '__main__':

    # Train and evaluate decoder
    iterations = 2000

    gammas = [0.99, 0.99]
    training_data = []

    for it in range(iterations):
        if it % 100 == 0:
            print(f'Iteration {it} of {iterations}')
        mdp = generate_mdp()
        multi_gamma_q_values = generate_multi_gamma_qvalues(mdp, gammas, episodes=1000)
        
        # Extract and append training data for state s1
        input_data_s1, label_s1 = prepare_training_data_for_state(multi_gamma_q_values, 's1')
        training_data.append((input_data_s1, label_s1))

        # Extract and append training data for state s9
        input_data_s9, label_s9 = prepare_training_data_for_state(multi_gamma_q_values, 's9')
        training_data.append((input_data_s9, label_s9))

    # Shuffle the combined training data
    random.shuffle(training_data)

    train_decoder(training_data, gammas=gammas, learning_rate=0.001)

    ####################

    # Produce Figure that computes performance manually by identifying larger gamma
    iterations = 1000
    gammas = [0.6, 0.7, 0.8, 0.9,0.99]
    fractions = []

    for gamma in gammas:
        print(gamma)
        for _ in range(iterations):
            mdp = generate_mdp()
            memory = perform_trajectories(mdp)
            q_values = q_learning(mdp, memory, gamma=gamma)

            s1_up_larger = 1 if q_values['s1']['up'] > q_values['s1']['down'] else 0
            s9_up_larger = 1 if q_values['s9']['up'] > q_values['s9']['down'] else 0

            fractions.append({'gamma': gamma, 'state': 's1', 'up_larger': s1_up_larger})
            fractions.append({'gamma': gamma, 'state': 's9', 'up_larger': s9_up_larger})

    df = pd.DataFrame(fractions)

    # calculate the mean and standard deviation
    df_summary = df.groupby(['gamma', 'state']).agg(['mean', 'std'])['up_larger'].reset_index()

    # calculate the desired standard deviation
    df_summary['half_std'] = df_summary['std'] / 4

    # melt the data frame to long format
    df_melt = df_summary.melt(id_vars=['gamma', 'state'], value_vars=['mean', 'half_std'])

    # draw lineplot with ci=None and use fill_between to add shaded errorbars manually
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(3, 3))
    g = sns.lineplot(data=df_melt[df_melt['variable'] == 'mean'], x='gamma', y='value', hue='state',marker='o', ax=ax,palette=['C1', 'C0'])
    for state, color in zip(['s1', 's9'], ['C1', 'C0']):
        ax.fill_between(df_summary['gamma'][df_summary['state'] == state], 
                        df_summary['mean'][df_summary['state'] == state] - df_summary['half_std'][df_summary['state'] == state], 
                        df_summary['mean'][df_summary['state'] == state] + df_summary['half_std'][df_summary['state'] == state], 
                        color=color, alpha=0.3)

    for line in ax.lines:
        line.set_markeredgecolor("none")

    plt.xlabel('Gamma')
    plt.ylabel('Performance')
    plt.legend().remove()  # remove the legend
    plt.savefig("mdp_myopic.svg", bbox_inches="tight")
    plt.show()



    df = pd.DataFrame(fractions)

    df_summary = df.groupby(['gamma', 'state']).agg(['mean', 'std'])['up_larger'].reset_index()
    df_summary['std_half'] = df_summary['std'] / 4

    sns.set_theme(style="white")
    g = sns.pointplot(
        data=df_summary, x='gamma', y='mean', hue='state', 
        capsize=.2, palette=['C1', 'C0'],
    )

    for idx, row in df_summary.iterrows():
        plt.errorbar(
            row['gamma'], row['mean'], yerr=row['std_half'],
            fmt='none', ecolor='black', capsize=3, capthick=1, elinewidth=1
        )

    g.set(xlabel='Gamma', ylabel='Performance')
    plt.show()