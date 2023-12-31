import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Method to perform td learning in the two experiments
def td_learning(num_states, num_backups, alpha=0.02, gamma = 0.9, reward_magnitude=1):
    values = np.zeros(num_states)
    learned_values = []
    for _ in range(num_backups):
        state = 0
        for t in range(num_states - 1):
            reward = reward_magnitude if t == num_states - 2 else 0  # Reward is 1 in the final state.
            if t==num_states - 2:
                values[state] += alpha * (reward  - values[state])
            else:
                values[state] += alpha * (reward + gamma* values[state + 1] - values[state])
            state += 1
        learned_values.append(values[0])  # Store the value of the initial state after each backup
    return learned_values

# Parameters
max_backups = 200  # Maximum number of TD-backups
short_mdp_states = 2  # 10 transitions, 11 states
long_mdp_states = 4  # 40 transitions, 41 states

# TD-Learning for two MDPs for varying number of backups
short_mdp_values = [td_learning(short_mdp_states, backups+1)[-1] for backups in range(max_backups)]
long_mdp_values = [td_learning(long_mdp_states, backups+1)[-1] for backups in range(max_backups)]

# Plotting the learned values
import matplotlib.cm as cm
color_map = cm.get_cmap('cool')
short_mdp_color = color_map(300)  # color at value 1 from 'cool' cmap
long_mdp_color = color_map(30)  # color at value 0.05 from 'cool' cmap

# Plotting the learned values
fig, ax = plt.subplots(figsize=(4, 3))

# Plot for short MDP
ax.plot(range(max_backups), short_mdp_values, label=f'Short MDP ({short_mdp_states - 1} transitions)', color=short_mdp_color)
ax.tick_params(axis='x', direction='in', colors=short_mdp_color)

# Create a secondary X-axis for the long MDP with inverted x-axis and different color
ax2 = ax.twiny()
ax2.plot(range(max_backups-1, -1, -1), long_mdp_values, label=f'Long MDP ({long_mdp_states - 1} transitions)', color=long_mdp_color)
ax2.tick_params(axis='x', direction='in', colors=long_mdp_color)

# Manually setting the tick labels for the long MDP
ax2.set_xticks(np.linspace(0, max_backups, max_backups//25+1))
ax2.set_xticklabels([str(int(label)) for label in np.linspace(max_backups, 0, max_backups//25+1)])

# Setting labels and legend
ax.set_xlabel('Number of TD-backups for Short Trajectory', color=short_mdp_color)
ax2.set_xlabel('Number of TD-backups for Long Trajectory)', color=long_mdp_color)
ax.set_ylabel('Value of Initial State')

plt.show()
fig.savefig('figures/my_figure.svg', format='svg')


gammas = [0.6, 0.9, 0.99]

fig, ax = plt.subplots(figsize=(4, 3))

# Parameters
max_backups = 100  # Maximum number of TD-backups
short_mdp_states = 3  # 10 transitions, 11 states
long_mdp_states = 11  # 40 transitions, 41 states
reward_magn=1

# Plot for Short MDP
for gamma, color in zip(gammas, ['cyan', 'blue', 'purple']):
    short_mdp_values = [td_learning(short_mdp_states, backups+1, gamma=gamma, alpha=0.1)[-1] for backups in range(max_backups)]
    ax.plot(range(max_backups), short_mdp_values, label=f'Gamma = {gamma}', color=short_mdp_color)
ax.set_title(f'Short MDP ({short_mdp_states - 1} transitions)')
ax.set_xlabel('Number of TD-backups')
ax.set_ylabel('Value of Initial State')
ax.legend(loc='upper center')
plt.show()
fig.savefig('figures/my_figure.svg', format='svg')

fig, ax = plt.subplots(figsize=(4, 3))

# Plot for Long MDP
for gamma, color in zip(gammas, ['magenta', 'red', 'orange']):
    long_mdp_values = [td_learning(long_mdp_states, backups+1, gamma=gamma,alpha=0.1)[-1] for backups in range(max_backups)]
    ax.plot(range(max_backups), long_mdp_values, label=f'Gamma = {gamma}', color=long_mdp_color)
ax.set_title(f'Long MDP ({long_mdp_states - 1} transitions)')
ax.set_xlabel('Number of TD-backups')
ax.set_ylabel('Value of Initial State')
ax.legend(loc='upper center')

plt.show()
fig.savefig('figures/my_figure.svg', format='svg')

