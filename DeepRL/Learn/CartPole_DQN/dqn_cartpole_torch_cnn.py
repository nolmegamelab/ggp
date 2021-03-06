import gym
import matplotlib 
import numpy as np 
import matplotlib.pyplot as plt 
import collections 
import itertools 
from PIL import Image
import random

import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as T 

env = gym.make('CartPole-v0').unwrapped

is_iptyhon = 'inline' in matplotlib.get_backend()
if is_iptyhon:
    from IPython import display 

plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object): 

    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    
class DQN(nn.Module):

    def __init__(self, h, w, outputs):

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) -1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # view changes indexing only (dimension and size)
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(), 
                    T.Resize(40, interpolation=Image.CUBIC), 
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    screen = env.render(mode='rgb_array').transpose(2, 0, 1)
    _, screen_height, screen_width = screen.shapes
    screen = screen[:, int(screen_height * 0.4):int(screen_height*0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, 
                            cart_location + view_width // 2) 

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

env.reset()
plt.figure() 
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), 
            interpolation='none')
plt.title('Example extracted screen')
plt.show()

BATCH_SIZE = 128 
GAMMA = 0.999 
EPS_START = 0.9 
EPS_END = 0.05 
EPS_DECAY = 200 
TARGET_UPDATE = 10 

init_screen = get_screen() 
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device) 
target_net.load_state_dict(policy_net.state_dice())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')

