import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

from skimage.color import rgb2gray
from skimage.transform import resize
import logger

# from distper 
class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, action_size, state_size, device):
        """초기화."""
        super(DQN, self).__init__()

        self.device = device
        self.input_shape = state_size
        self.num_actions = action_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(self.input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
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
        conv = self.conv(x)
        conv_out = conv.view(x.size()[0], -1)
        return self.fc(conv_out)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            state = state.unsqueeze(0)          # make it in a batch format [1, 4, 84, 84]
            state = state.to(self.device)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.cpu().numpy()[0]




# 브레이크아웃 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size, memory=None, state_size=(4, 84, 84)):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        # 0.6 : 한번 훈련 후 지정 
        # 0.1 : 꽤 잘 플레이 하면 지정 
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon_start, self.epsilon_end = 0.8, 0.00001
        self.epsilon = self.epsilon_start
        self.exploration_steps = 90000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.batch_size = 32 
        self.train_start = 10000
        self.update_target_rate = 10000

        # 리플레이 메모리
        if memory is None:
            self.memory = deque(maxlen=500000)
        else: 
            self.memory = memory
        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 10
        self.device = 'cuda'
        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size, self.device)
        self.model.to(self.device)
        self.target_model = DQN(action_size, state_size, self.device)
        self.target_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        load_model = True

        if load_model:
            self.model.load_state_dict(torch.load("./save_model/model_torch.pth"))

        # 타깃 모델 초기화
        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0

        self.writer = tf.summary.create_file_writer('summary/breakout_dqn')
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        return self.model.act(state, self.epsilon)

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

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
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        # 이 부분은 논문과 다르다 - 최신에서 샘플링. 
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in batch],
                           dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch],
                                dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        states_float = np.array(states).astype(np.float32) 
        next_states_float = np.array(next_states).astype(np.float32) 
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # convert back to float from uint8 
        states_tensor = torch.FloatTensor(states_float).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_float).to(self.device)
        actions_tensor = torch.from_numpy(actions)
        actions_tensor = actions_tensor.type(torch.int64).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor)
            next_q_a_values = next_q_values.max(1)[0]
            # rewards가 이전 step의 보상을 포함하고 있어 최종 보상만 감쇄 반영한다. 
            expected_q_a_values = rewards_tensor + self.discount_factor * next_q_a_values * (1 - dones_tensor)
            expected_q_a_values = expected_q_a_values.to(self.device).unsqueeze(1)

        q_values = self.model(states_tensor)
        q_a_values = q_values.gather(-1, actions_tensor)

        td_error = torch.abs(expected_q_a_values - q_a_values)
        # Huber loss is very important for the convergence of breakout play
        loss = torch.nn.SmoothL1Loss()(expected_q_a_values, q_a_values)

        priorities = (td_error + -1e-6).clone().detach().cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        self.avg_loss += loss.detach().cpu().numpy()


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    # rgb2gray에서 float 형으로 변경
    processed_observe = resize(rgb2gray(observe), (84, 84), mode='constant') 
    return processed_observe


if __name__ == "__main__":
    memory = deque(maxlen=120000)

    logger = logger.Logger('atari_torch.log')

    for loop in range(0, 1000):
        # 환경과 DQN 에이전트 생성
        #env = gym.make('BreakoutDeterministic-v4', render_mode='human')
        env = gym.make('BreakoutDeterministic-v4')
        render = False

        agent = DQNAgent(action_size=3, memory=memory)

        global_step = 0
        score_avg = 0
        score_max = 0

        # 불필요한 행동을 없애주기 위한 딕셔너리 선언
        action_dict = {0:1, 1:2, 2:3, 3:3}

        num_episode = 100000
        for e in range(num_episode):
            done = False
            dead = False

            step, score, start_life = 0, 0, 5
            # env 초기화
            observe = env.reset()

            # 랜덤으로 뽑힌 값 만큼의 프레임동안 움직이지 않음
            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ = env.step(1)

            # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
            state = pre_processing(observe)
            history = np.stack((state, state, state, state), axis=0) # (4, 84, 84)

            while not done:
                if render:
                    env.render()
                global_step += 1
                step += 1

                # 바로 전 history를 입력으로 받아 행동을 선택
                action, q_values = agent.get_action(history)
                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                real_action = action_dict[action]

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action, real_action, dead = 0, 1, False

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                observe, reward, done, info = env.step(real_action)
                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(observe)
                next_history = np.zeros((4, 84, 84)) 
                next_history[0, :, :] = history[1, :, :]
                next_history[1, :, :] = history[2, :, :]
                next_history[2, :, :] = history[3, :, :]
                next_history[3, :, :] = next_state

                agent.avg_q_max += np.max(q_values)

                if start_life > info['lives']:
                    dead = True
                    reward = -1
                    start_life = info['lives']

                score += reward
                reward = np.clip(reward, -1., 1.)
                # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
                agent.append_sample(history, action, reward, next_history, dead)

                # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
                if len(agent.memory) >= agent.train_start:
                    #for i in range(0, 4):
                    agent.train_model()
                    # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()

                if dead:
                    history = np.stack((next_state, next_state,
                                        next_state, next_state), axis=0)
                else:
                    history = next_history

                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    if global_step > agent.train_start:
                        agent.draw_tensorboard(score, step, e)

                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    log = "loop: {:5d} | ".format(loop)
                    log += "episode: {:5d} | ".format(e)
                    log += "score: {:4.1f} | ".format(score)
                    log += "score max : {:4.1f} | ".format(score_max)
                    log += "score avg: {:4.1f} | ".format(score_avg)
                    log += "memory length: {:5d} | ".format(len(agent.memory))
                    log += "epsilon: {:.3f} | ".format(agent.epsilon)
                    log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                    log += "avg loss : {:3.4f}".format(agent.avg_loss / float(step))
                    logger.info(log)

                    agent.avg_q_max, agent.avg_loss = 0, 0

                # start from beginning
            if agent.epsilon < 0.001:
                break

            # 모델 저장
            if (e+1) % 50 == 0:
                torch.save(agent.model.state_dict(), "./save_model/model_torch.pth")
                print('model saved\n')
