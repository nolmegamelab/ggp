"""
Module for DQN Model in Ape-X.
"""
import random
import numpy as np
import torch
import torch.nn as nn

# from distper 
class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, env, device):
        """초기화."""
        super(DQN, self).__init__()

        self.device = device
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(self.input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.01)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """전방 연쇄."""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = state.unsqueeze(0) # make it in a batch format [1, 1, 84, 84]
            state = state.to(self.device)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.cpu().numpy()[0]


class DuelingDQN(nn.Module):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, device='cpu'):
        super(DuelingDQN, self).__init__()
        self.device = device
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.flatten = Flatten()

        conv2d_1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        init(conv2d_1)
        conv2d_1.weight.data.fill_(0.0)
        conv2d_1.bias.data.fill_(0.0)

        conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        init(conv2d_2)
        conv2d_2.weight.data.fill_(0.0)
        conv2d_2.bias.data.fill_(0.0)

        conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        init(conv2d_3)
        conv2d_3.weight.data.fill_(0.0)
        conv2d_3.bias.data.fill_(0.0)

        self.features = nn.Sequential(
            conv2d_1, 
            nn.ReLU(), 
            conv2d_2, 
            nn.ReLU(), 
            conv2d_3, 
            nn.ReLU()
        )

        linear_1 = nn.Linear(self._feature_size(), 512)
        init(linear_1)
        linear_1.weight.data.fill_(0.0)
        linear_1.bias.data.fill_(0.0)

        linear_2 = nn.Linear(512, self.num_actions)
        init(linear_2)
        linear_2.weight.data.fill_(0.0)
        linear_2.bias.data.fill_(0.0)

        self.advantage = nn.Sequential( 
            linear_1,
            nn.ReLU(),
            linear_2
        )

        linear_3 = nn.Linear(self._feature_size(), 512)
        init(linear_3)
        linear_3.weight.data.fill_(0.0)
        linear_3.bias.data.fill_(0.0)

        linear_4 = nn.Linear(512, 1)
        init(linear_4)
        linear_4.weight.data.fill_(0.0)
        linear_4.bias.data.fill_(0.0)

        self.value = nn.Sequential(
            linear_3,
            nn.ReLU(),
            linear_4
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = state.unsqueeze(0) # make it in a batch format [1, 1, 84, 84]
            state = state.to(self.device)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.cpu().numpy()[0]


class Flatten(nn.Module):
    """
    Simple module for flattening parameters
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


def init_(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init(module):
    return init_(module,
                 nn.init.orthogonal_,
                 lambda x: nn.init.constant_(x, 0),
                 nn.init.calculate_gain('relu'))
