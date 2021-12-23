import collections 
import numpy as np
import multiprocessing
import os
import pickle
import tensorboardX 
import time
import torch
import traceback

import buffer
import env_wrappers
import model
import mq
import msgpack
import msgpack_numpy as msgnum
import options
import utils

class ExploreStepStorage:
    """
    Storage for actors to support multi-step learning and efficient priority calculation.
    Saving Q values with experiences enables td-error priority calculation
    without re-calculating Q-values for each state.
    """
    def __init__(self, opts, n_step_count=100):
        """
            n_steps - n_steps reward calculation steps
            capacity 
        """
        self.opts = opts
        self.n_step = opts.n_step
        self.capacity = n_step_count * self.n_step
        self.step_deque = collections.deque(maxlen=self.capacity)
        self.send_deque = collections.deque(maxlen=self.capacity)
        self.gamma = self.opts.gamma
        self.memory = buffer.ReplayBuffer(self.opts.learner_storage_capacity) 
                            #self.opts.alpha)

    def add(self, state, reward, action, done, q_values):
        '''
        calculates n-step reward  
        '''
        # when n_step is accumulated, we calculate td_error and prepares to send
        if len(self.step_deque) == self.n_step:
            target_step = self.step_deque[0]
            reward_n_step = self._multi_step_reward(reward)
            next_q_a_max = q_values.max()
            reward_q_a_value = reward_n_step + (self.gamma ** (self.n_step+1)) * next_q_a_max * (1 -done) 
            td_error = reward_q_a_value - target_step['q_a_value']
            next_state = self.step_deque[1]['state']

            self.memory.add(
                target_step['state'], 
                target_step['action'], 
                target_step['reward'],
                next_state, 
                target_step['done'] 
            )

            self.step_deque.popleft()

        # put the new step into the step queue 
        self.step_deque.append({
            'state': state, 
            'reward': reward,
            'action': action,
            'done': done,
            'q_a_value': q_values[action]            
        })

    def sample(self):
        if len(self.memory) > self.opts.batch_size * 100:
            #return self.memory.sample(self.opts.batch_size, self.opts.beta)
            return self.memory.sample(self.opts.batch_size)
        else: 
            return None

    def _multi_step_reward(self, reward):
        ret = 0.
        for i in range(0, self.n_step):
            ret += self.step_deque[i]['reward'] * (self.gamma ** i)
        ret += reward * (self.gamma ** self.n_step)
        return ret

    def __len__(self):
        return len(self.states)

class Actor:
    """
    Actor explores the environment and accumulates rollout steps 
        - ExploreStepStorage passes steps into a PriorityReplayBuffer
        - Then, Actor samples from the buffer to train
    """

    def __init__(self, actor_index, actor_count, opts):
        self.actor_index = actor_index
        self.actor_count = actor_count 
        self.opts = opts

    def prepare(self):
        '''
        prepares exploring the env
            - setup atari environment
            - create the NN model 
            - receives the initial NN parameters 
        '''
        msgnum.patch()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # setting the number of threads to 1 solves high cpu utilization
        torch.set_num_threads(1)

        self.writer = tensorboardX.SummaryWriter(comment="-{}-actor{}".format(self.opts.env, self.actor_index))

        self.env = env_wrappers.make_atari(self.opts.env)
        self.env = env_wrappers.wrap_atari_dqn(self.env, self.opts)

        seed = self.opts.seed + self.actor_index
        utils.set_global_seeds(seed, use_torch=True)

        self.env.seed(seed)

        # setup learning environment 
        # - check mq 
        # - check learner 
        # - receive initial model parameters 

        self.model = model.DQN(self.env, self.device)
        self.model.load_state_dict(torch.load("model_local.pth"))
        self.target_model = model.DQN(self.env, self.device)
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.storage = ExploreStepStorage(self.opts, 1000)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.opts.learning_rate)

    def explore(self, min_epsilon=0.001):
        """
        explores the environment and forward steps (transitions) to the collector
        """
        div = max(1, self.actor_count - 1)
        self.epsilon = self.opts.epsilon_base ** (1 + self.actor_index / div * self.opts.epsilon_alpha)
        self.epsilon_decay = self.opts.epsilon_decay
        episode_reward, episode_length, episode_idx, param_age = 0, 0, 0, 0
        state = self.env.reset()
        sum_q_value = 0
        max_q_value = 0
        sum_loss = 0
        learning_loop_count = 0

        while True:
            state = np.array(state)
            action, q_values = self.model.act(torch.FloatTensor(state), self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.storage.add(state, reward, action, done, q_values)

            if self.opts.env_render: 
                self.env.render()

            sum_q_value += q_values[action]
            max_q_value = max(max_q_value, q_values[action])
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.epsilon = max(0, self.epsilon-self.epsilon_decay)
            self.epsilon = max(0.001, self.epsilon)

            if done or episode_length == self.opts.max_episode_length:
                self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                print(f"episode: {episode_idx:5.0f}, epsilon: {self.epsilon:1.5f}, reward: {episode_reward:5.1f}, length: {episode_length:5d}, age: {param_age:5d}, q: {sum_q_value/episode_length:4.3f}, max q: {max_q_value:4.3f}, loss: {sum_loss / episode_length:4.3f}")
                episode_reward = 0
                episode_length = 0
                episode_idx += 1
                sum_q_value = 0
                sum_loss = 0
                state = self.env.reset()

            batch = self.storage.sample()
            if batch is not None:
                loss, priorities = self._forward(batch)
                self._backward(loss)
                sum_loss += loss

            if learning_loop_count % self.opts.target_update_interval == 0:
                print("Updating Target Network..")
                param_age += 1
                self.target_model.load_state_dict(self.model.state_dict())

            if learning_loop_count % self.opts.save_interval == 0:
                print("Saving Model..")
                torch.save(self.model.state_dict(), "model_local.pth")

            time.sleep(0.001)

            learning_loop_count += 1

            if self.epsilon <= min_epsilon:
                break

    def finish(self):
        pass

    def _forward(self, batch):
        states, actions, rewards, next_states, dones = batch

        states_float = np.array(states).astype(np.float32) / 255.0
        next_states_float = np.array(next_states).astype(np.float32) / 255.0
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
            expected_q_a_values = rewards_tensor + self.opts.gamma * next_q_a_values * (1 - dones_tensor)
            expected_q_a_values = expected_q_a_values.to(self.device)

        q_values = self.model(states_tensor)
        q_a_values = q_values.gather(-1, actions_tensor).squeeze(1)

        #  후버로스 계산
        td_error = torch.abs(expected_q_a_values - q_a_values)
        #quadratic_part = torch.clip(td_error, 0.0, 1.0)
        #linear_part = td_error - quadratic_part

        #loss = 0.5 * quadratic_part ** 2 + linear_part
        #loss = loss.mean()
        loss = torch.nn.MSELoss()(expected_q_a_values, q_a_values)
        #weights_tensor = torch.from_numpy(weights).to(self.device)
        #loss = (loss * weights_tensor).mean()

        priorities = (td_error + -1e-6).clone().detach().cpu().numpy()
        return loss, priorities


    def _backward(self, loss):
        """
        Update parameters with loss
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.opts.max_grad)
        self.optimizer.step()
        # NOTE: removed total norm calculation

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    multiprocessing.set_start_method('spawn')
    opts = options.Options()
    actor = Actor(1, 1, opts)
    try:
        actor.prepare()
        for i in range(0, 10000):
            print(f'loop: {i}')
            actor.explore(0.002)
    except Exception as e:
        print(f'exception: {e}')
        traceback.print_exc()
    finally:
        actor.finish()
