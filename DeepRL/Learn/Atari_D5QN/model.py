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

    def __init__(self, env, action_map, device):
        """초기화."""
        super(DQN, self).__init__()

        self.device = device
        self.input_shape = env.shape
        self.num_actions = env.action_space.n
        self.action_map = action_map

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
        #self.conv.apply(self.init_weights)
        #self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
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
            state = state.unsqueeze(0)          # make it in a batch format [1, 1, 84, 84]
            state = state.to(self.device)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return self.action_map[action], q_values.cpu().numpy()[0]


