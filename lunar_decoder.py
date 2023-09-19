# import sys
import collections

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score

DEVICE = "cpu"

DEFAULT_ENV_NAME = "LunarLander-v2"

WHICH_LAYER = "FC"  # input #output

TARGET_LENGTH = 50
ITERATIONS = 100_000
NUM_TRAINING_SAMPLES = 10_000
TEST_SIZE = 2_000

BATCH_SIZE = 32
Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


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


class Classifier(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512), nn.ReLU(), nn.Linear(512, output_shape)
        )

    def forward(self, x):
        return self.fc(x)


def decode_temporal_evolution(net, net_sim, plot_test=False):

    device = "cpu"

    env = gym.make(DEFAULT_ENV_NAME)

    buffer = ExperienceBuffer(NUM_TRAINING_SAMPLES)

    print("Preparing training pipeline...")
    buffer.fill_buffer(net=net_sim, env=env, size=NUM_TRAINING_SAMPLES)

    print("Buffer filled")
    states = np.array(
        [buffer.buffer[start].state for start in range(0, len(buffer) - TARGET_LENGTH)]
    )
    rewards = np.array(
        [
            np.array(
                [
                    buffer.buffer[t].reward
                    for t in np.arange(start, start + TARGET_LENGTH)
                ]
            )
            for start in range(0, len(buffer) - TARGET_LENGTH)
        ]
    )

    rewards[np.abs(rewards) < 5] = 0
    print("States and rewards collected")
    running_loss = 0.0

    print("Ready to train classifier..")

    criterion = nn.MSELoss()

    # new_index=[index for index in range(1,len(rewards)-1) if np.prod(rewards[index,:]==0)==0] #indeces without all zero rewards
    new_index = range(1, len(rewards) - 1)

    if WHICH_LAYER == "input":
        input_size = env.observation_space.shape[0]
    elif WHICH_LAYER == "FC":
        input_size = 512
    elif WHICH_LAYER == "output":
        input_size = env.action_space.n * NUM_GAMMAS

    classifier = Classifier(input_size, TARGET_LENGTH).to(
        device
    )  # frames:28224   conv:3136   output:num_actiosns*num_h*num_gammas
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    train_index = np.random.choice(new_index, len(new_index) - TEST_SIZE, replace=False)
    test_index = list(set(new_index).symmetric_difference(set(train_index)))

    # train_index=np.arange(len(rewards)-TEST_SIZE)
    # test_index=np.arange(len(rewards)-TEST_SIZE,len(rewards))

    ############################# TRAIN #######################################
    for i in range(0, ITERATIONS):
        start = np.random.choice(train_index, BATCH_SIZE)
        indices = np.array([np.arange(start, start + TARGET_LENGTH) for start in start])

        frame = torch.tensor(np.array(states[start, :], copy=False)).to(device)

        with torch.no_grad():

            if WHICH_LAYER == "input":
                pre_input_classifier = frame.to("cpu").numpy()
            elif WHICH_LAYER == "FC":
                pre_input_classifier = (
                    net.fc2(net.fc1(frame)).detach().to("cpu").numpy()
                )
            elif WHICH_LAYER == "output":
                pre_input_classifier = net(frame).detach().to("cpu").numpy()

            input_classifier = torch.tensor(
                [pre_input_classifier[i, :].flatten() for i in range(0, BATCH_SIZE)]
            ).to(device)
            target_classifier = torch.tensor(rewards[start, :]).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(input_classifier).to(device)
        loss = criterion(outputs, target_classifier.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 2000 mini-batches
            print("iteration: %5d,  training loss: %.3f" % (i + 1, running_loss / 1000))
            running_loss = 0.0

    print("Finished Training")

    ############################# TEST #######################################
    print("Starging Testing...")

    running_loss = 0.0
    losses = []

    for i in range(0, 20):
        print(i)
        start = np.random.choice(test_index, 999)

        frame = torch.tensor(np.array(states[start, :], copy=False)).to(device)

        #
        with torch.no_grad():
            if WHICH_LAYER == "input":
                pre_input_classifier = frame.to("cpu").numpy()
            elif WHICH_LAYER == "FC":
                pre_input_classifier = (
                    net.fc2(net.fc1(frame)).detach().to("cpu").numpy()
                )
            elif WHICH_LAYER == "output":
                pre_input_classifier = net(frame).detach().to("cpu").numpy()

            input_classifier = torch.tensor(
                [pre_input_classifier[i, :].flatten() for i in range(0, 999)]
            ).to(device)
            target_classifier = torch.tensor(rewards[start, :]).to(device)

            outputs = classifier(input_classifier).to(device)

        loss = r2_score(
            savgol_filter(np.array(target_classifier.to("cpu").detach()), 5, 1),
            savgol_filter(np.array(outputs.to("cpu").detach()), 5, 1),
        )

        losses.append(loss.item())
    print("R^2 score: ", np.mean(losses))

    if plot_test:
        plt.figure()
        plt.plot(
            savgol_filter(np.array(target_classifier.to("cpu").detach()), 5, 1),
            savgol_filter(np.array(outputs.to("cpu").detach()), 5, 1),
            ".",
        )
        plt.title(
            r2_score(
                np.array(target_classifier.to("cpu").detach()),
                np.array(outputs.to("cpu").detach()),
            )
        )
        plt.show()

    return np.mean(losses)


if __name__ == "__main__":

    all_decoding_paths = [
        "lunar_multi_gamma_0.pt",
        "lunar_multi_gamma_1.pt",
        "lunar_multi_gamma_2.pt",
    ]
    sim_path = "lunar_multi_gamma_0.pt"

    decoding_performance = []
    for net_path in all_decoding_paths:
        print(net_path)
        NUM_GAMMAS = 25
        net = torch.load(net_path)
        NUM_GAMMAS = 1
        net_sim = torch.load(sim_path)
        NUM_GAMMAS = 25

        decoding_performance.append(decode_temporal_evolution(net, net_sim))

    print(("MEAN PERFORMANCE: ", np.mean(decoding_performance)))
