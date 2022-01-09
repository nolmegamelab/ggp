import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque

import replay
import logger
import env as atari
import renderer as scene
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

# from distper 
class ActorCritic(nn.Module):
    """Deep Q-Network."""

    def __init__(self, action_size, state_size, device):
        """초기화."""
        super(ActorCritic, self).__init__()

        self.device = device
        self.input_shape = state_size
        self.num_actions = action_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(self.input_shape)
        self.lin = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        # actor: policy
        self.action_head= nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, self.num_actions)
        )        

        # critic: value
        self.value_head= nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1) 
        )

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):       
        conv = self.conv(x)
        conv_out = conv.view(x.size()[0], -1)
        lin = self.lin(conv_out)

        action_prob = F.softmax(self.action_head(lin), dim=-1)
        state_value = self.value_head(lin)

        return action_prob, state_value

    def act(self, state):
        """
        Return action, max_q_value for given state
        """

        # with no_grad() removed to require grad
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        probs, state_value = self.forward(state)

        m = torch.distributions.Categorical(probs)
        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item(), state_value.item()

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


# DUEL PER DQN Agent
class ActorCriticAgent:
    def __init__(self, action_size, state_size=(4, 84, 84)):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 3e-4

        self.device = 'cuda'
        # 모델 생성
        self.model = ActorCritic(action_size, state_size, self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        load_model = True

        if load_model:
            self.model.load_state_dict(torch.load("./save_model/model_ac_breakout.pth"))

        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/breakout_ac')
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        return self.model.act(state)

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode',
                              self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode',
                              self.avg_loss / float(step), step=episode)

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):

        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.discount_factor * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            value = value.squeeze(1)
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]
        self.avg_loss += loss.cpu().detach().item()

def explore():

    log = logger.Logger('ac_atari.log')

    for epoch in range(0, 1000):
        log.info('------------------------------------------------')
        log.info(f'begin loop {epoch}')
        log.info('------------------------------------------------')
        # 환경과 DQN 에이전트 생성

        #env = gym.make('BreakoutDeterministic-v4', render_mode='human')
        env = atari.Environment(rom_file='roms/Breakout.bin', frame_skip=4, num_frames=4, 
                            no_op_start=5, dead_as_end=True)

        render = True

        if render:
            renderer = scene.Renderer(84, 84, 84*4, 84, title='Break! out')

        agent = ActorCriticAgent(action_size=4)

        global_step = 0
        score_avg = 0
        score_max = 0
        value_avg = 0
        value_max = 0
        num_episode = 1000

        for e in range(num_episode):
            done = False
            dead = False

            step, score = 0, 0

            # env 초기화
            state = env.reset()
            next_state = state

            while not done:
                global_step += 1
                step += 1

                state = next_state
                
                if render: 
                    hs = state * 255
                    for i in range(4):
                        h = hs[i]
                        mem = np.stack([h, h, h], axis=2)
                        renderer.render(i*84, 0, mem)
                    renderer.swap()
                
                # 바로 전 history를 입력으로 받아 행동을 선택
                action, state_value = agent.get_action(state)

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action = 1

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                observe, reward, dead, done = env.step(action)
                # 각 타임스텝마다 상태 전처리
                next_state = observe

                if dead:
                    reward = -5

                score += reward
                reward = np.clip(reward, -5., 1.)

                agent.model.rewards.append(reward)
                value_max = max(value_max, state_value)
                value_avg = 0.9 * value_avg + 0.1 * state_value

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    l = "epoch: {:5d} | ".format(epoch)
                    l += "episode: {:5d} | ".format(e)
                    l += "score: {:4.1f} | ".format(score)
                    l += "score max : {:4.1f} | ".format(score_max)
                    l += "score avg: {:4.1f} | ".format(score_avg)
                    l += "value avg: {:4.1f} | ".format(value_avg)
                    l += "value max: {:4.1f} | ".format(value_max)
                    l += "avg loss : {:3.4f}".format(agent.avg_loss / float(step))
                    log.info(l)

                    agent.avg_q_max, agent.avg_loss = 0, 0

            agent.train_model()

            # 모델 저장
            if (e+1) % 50 == 0:
                torch.save(agent.model.state_dict(), "./save_model/model_ac_breakout.pth")
                log.info(f'model saved. global_step: {global_step}')

if __name__ == "__main__":

    explore()