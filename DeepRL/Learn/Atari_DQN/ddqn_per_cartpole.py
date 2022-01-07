import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import gym
import os
import random
import numpy as np
import tensorflow as tf
from collections import deque

import replay
import logger
import env as atari
import renderer as scene

# from distper 
class DuelDQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, action_size, state_size, device):
        """초기화."""
        super(DuelDQN, self).__init__()

        self.device = device
        self.input_shape = state_size
        self.num_actions = action_size

        self.lin = nn.Sequential(
            nn.Linear(self.input_shape, 64),
            nn.ReLU(),
        )

        self.state_value = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(), 
            nn.Linear(256, 1)
        )        

        self.action_value = nn.Sequential(
            nn.Linear(64, 256), 
            nn.ReLU(), 
            nn.Linear(256, self.num_actions)
        )

    def forward(self, x):       
        lin = self.lin(x)

        action_value = self.action_value(lin)
        state_value = self.state_value(lin)
        action_score_centered = action_value - action_value.mean(dim=1, keepdim=True)

        q = state_value + action_score_centered
        return q

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = torch.tensor(state, device=self.device)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.cpu().numpy()[0]

# DUEL PER DQN Agent
class DuelDQNAgent:
    def __init__(self, action_size, memory=None, state_size=4, eps_start=1):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        # 0.6 : 한번 훈련 후 지정 
        # 0.1 : 꽤 잘 플레이 하면 지정 
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon_start, self.epsilon_end = eps_start, 0.00001
        self.epsilon = self.epsilon_start
        self.exploration_steps = 10000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps

        self.beta_start, self.beta_end = 0.4, 1  # paper: 0.4 -> 1
        self.beta_annealing_step = self.beta_end - self.beta_start 
        self.beta_annealing_step /= self.exploration_steps 
        self.beta = self.beta_start

        self.batch_size = 32 
        self.train_start = 1000
        self.update_target_rate = 1000

        if memory is None:
            self.memory = replay.PriorityMemory(size=600000, state_shape=(600000, 84, 84))
        else:
            self.memory = memory

        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 10
        self.device = 'cuda'
        # 모델과 타깃 모델 생성
        self.model = DuelDQN(action_size, state_size, self.device)
        self.model.to(self.device)
        self.target_model = DuelDQN(action_size, state_size, self.device)
        self.target_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        load_model = False

        if load_model:
            self.model.load_state_dict(torch.load("./save_model/model_ddqn_cartpole.pth"))

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
    def add_experience(self, action, state, reward, dead):
        return self.memory.add_experience(action, state, reward, dead)

    def get_history(self, current_index):
        return self.memory.get_history(current_index)

    def get_frame(self, current_index):
        return self.memory.get_frame(current_index)

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

        if self.beta < self.beta_end:
            self.beta += self.beta_annealing_step

        batch = self.memory.get_minibatch(self.batch_size, self.beta)
        s_states, s_actions, s_rewards, s_nexts, s_dones, s_weights, s_indices = batch

        states = torch.tensor(s_states, device=self.device, dtype=torch.float32).squeeze(1)
        actions = torch.tensor(s_actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(s_rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(s_nexts, device=self.device, dtype=torch.float32).squeeze(1)
        dones = torch.tensor(s_dones, device=self.device, dtype=torch.int8)
        weights = torch.tensor(s_weights, device=self.device, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            double_q_values = self.model(next_states)
            next_q_values = self.target_model(next_states)

            best_next_action = torch.argmax(double_q_values, -1)
            next_q_a_values = next_q_values.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)
            # The above is different from DQN 
            #next_q_a_values = next_q_values.max(1)[0]

            # rewards가 이전 step의 보상을 포함하고 있어 최종 보상만 감쇄 반영한다. 
            expected_q_a_values = rewards + self.discount_factor * next_q_a_values * (1 - dones)
            expected_q_a_values = expected_q_a_values.to(self.device).unsqueeze(1)


        q_values = self.model(states)
        q_a_values = q_values.gather(-1, actions)

        td_error = torch.abs(expected_q_a_values - q_a_values)
        # Huber loss is very important for the convergence of breakout play
        losses = torch.nn.SmoothL1Loss(reduction='none')(expected_q_a_values, q_a_values)
        loss = torch.mean(weights * losses)

        # update weights
        self.optimizer.zero_grad()
        loss.backward()
        # max_norm is set to 0.5 (DQN is set to 10)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()

        priorities = (td_error + 1e-6).cpu().detach().numpy()
        self.memory.update_priorities(s_indices, priorities)
        self.avg_loss += loss.cpu().detach().item()

if __name__ == "__main__":
    logger = logger.Logger('atari_torch_ddqn_cartpole.log')
    memory = replay.PriorityMemory(size=650000, state_shape=(650000, 4))

    for loop in range(1, 1000):
        logger.info('------------------------------------------------')
        logger.info(f'begin loop {loop}')
        logger.info('------------------------------------------------')
        # 환경과 DQN 에이전트 생성

        #env = gym.make('BreakoutDeterministic-v4', render_mode='human')
        env = gym.make('CartPole-v1') 

        render = True

        agent = DuelDQNAgent(action_size=2, memory=memory, state_size=4, eps_start=1)

        global_step = 0
        score_avg = 0
        score_max = 0
        num_episode = 100000

        for e in range(num_episode):
            done = False
            dead = False

            step, score = 0, 0
            # env 초기화
            observe = env.reset()

            # 프레임을 전처리 한 후 4개의 상태를 쌓아서 입력값으로 사용.
            current_index = agent.add_experience(0, observe, 0, False)

            while not done:
                global_step += 1
                step += 1

                if current_index < 3: 
                    current_index = 3

                history = agent.get_frame(current_index)
                
                if render: 
                    env.render()
                
                # 바로 전 history를 입력으로 받아 행동을 선택
                action, q_values = agent.get_action(history)

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action = 1

                # 선택한 행동으로 환경에서 한 타임스텝 진행
                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                observe, reward, done, info = env.step(action)
                # 각 타임스텝마다 상태 전처리
                next_state = observe

                agent.avg_q_max += np.max(q_values)

                score += reward
                reward = np.clip(reward, -1., 1.)

                # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
                current_index = agent.add_experience(action, next_state, reward, dead)

                # 리플레이 메모리 크기가 정해놓은 수치에 도달한 시점부터 모델 학습 시작
                if agent.memory.get_size() >= agent.train_start:
                    #for i in range(0, 4):
                    agent.train_model()
                    # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
                    if global_step % agent.update_target_rate == 0:
                        agent.update_target_model()
                        logger.info(f'target updated. global_stp: {global_step}')

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
                    log += "memory length: {:5d} | ".format(agent.memory.get_size())
                    log += "epsilon: {:.3f} | ".format(agent.epsilon)
                    log += "beta: {:.3f} | ".format(agent.beta)
                    log += "max q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                    log += "avg loss : {:3.4f}".format(agent.avg_loss / float(step))
                    logger.info(log)

                    agent.avg_q_max, agent.avg_loss = 0, 0

                # start from beginning
            if agent.epsilon < 0.001:
                break

            # 모델 저장
            if (e+1) % 50 == 0:
                torch.save(agent.model.state_dict(), "./save_model/model_ddqn_cartpole.pth")
                logger.info(f'model saved. global_step: {global_step}')
