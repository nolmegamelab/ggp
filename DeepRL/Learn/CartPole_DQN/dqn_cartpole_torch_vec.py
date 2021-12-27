import sys
import gym
import pylab
import random
import numpy as np
from collections import deque 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

EPISODES = 1000

class DQN(nn.Module): 

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size 
        self.action_size = action_size
        self.model = nn.Sequential( 
            nn.Linear( 
                self.state_size,
                24 
            ), 
            nn.ReLU(), 
            nn.Linear(
                24, 
                24
            ),
            nn.ReLU(), 
            nn.Linear(
                24, 
                self.action_size
            )
        )

    def forward(self, x): 
        x = torch.FloatTensor(x)
        return self.model(x)

class DQNAgent: 

    def __init__(self, state_size, action_size):
        self.render = True
        self.load_model = True

        self.state_size = state_size
        self.action_size = action_size 

        self.discount_factor = 0.99 
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.99991
        self.epsilon_min = 0.001 
        self.batch_size =  128 
        self.train_start = 500 

        self.memory = deque(maxlen = 4000)

        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)

        if self.load_model: 
            self.model.load_state_dict(torch.load("./save_model/cartpole_torch_vec.pth"))

        self.update_target_model() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0].detach().numpy())

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self): 
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3] 
            dones.append(mini_batch[i][4])

        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

        target = self.model(states) 
        target_a = target.gather(-1, actions_tensor)
        target_val = self.target_model(next_states) 
        target_q_val = target_val.max(-1)[0].unsqueeze(1)
        expected_q = rewards_tensor + self.discount_factor * target_q_val * (1-dones_tensor) 

        loss = nn.MSELoss()(expected_q, target_a)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()

if __name__ == "__main__": 

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], [] 

    for e in range(EPISODES): 
        done = False 
        score = 0 

        state = env.reset() 
        state = np.reshape(state, [1, state_size])

        while not done: 
            if agent.render: 
                env.render() 

            action = agent.get_action(state) 
            next_state, reward, done, info = env.step(action) 
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100 

            agent.append_sample(state, action, reward, next_state, done) 

            if len(agent.memory) >= agent.train_start: 
                agent.train_model()

            score += reward 
            state = next_state 

            if done: 
                agent.update_target_model() 

                score = score if score == 500 else score + 100 

                scores.append(score) 
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/carpole_dqn.png")
                print("epsode: {} score: {} memory length: {} epsilon: {}".format(
                    e, score, len(agent.memory), agent.epsilon))
                
                #if np.mean(scores[-min(10, len(scores)):]) > 50: 
                torch.save(agent.model.state_dict(), "./save_model/cartpole_torch_vec.pth")
                #sys.exit()



                