import collections 
import numpy as np
import multiprocessing
import os
import pickle
import tensorboardX 
import time
import torch
import traceback

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
    def __init__(self, n_step, n_step_count=100, gamma=0.99):
        """
            n_steps - n_steps reward calculation steps
            capacity 
        """
        self.n_step = n_step
        self.capacity = n_step_count * n_step
        self.step_deque = collections.deque(maxlen=self.capacity)
        self.send_deque = collections.deque(maxlen=self.capacity)
        self.gamma = gamma

    def add(self, state, reward, action, done, q_values):
        '''
        calculates n-step reward  
        '''
        state = state._force()
        # when n_step is accumulated, we calculate td_error and prepares to send
        if len(self.step_deque) == self.n_step:
            target_step = self.step_deque[0]
            reward_n_step = self._multi_step_reward(reward)
            next_q_a_max = q_values.max()
            reward_q_a_value = reward_n_step + (self.gamma ** (self.n_step+1)) * next_q_a_max * (1 -done) 
            td_error = reward_q_a_value - target_step['q_a_value']
            next_state = self.step_deque[1]['state']

            # Ape-X paper compute priority while exploring, but it seems to be 
            # more accurate to calculate on the collector using PriorityReplayBuffer 
            # and samples on that priority. The collector, then, can send the batches 
            # to the Learner.
            # NOTE: tolist() converts a numpy array to a python list including element type conversion 
            self.send_deque.append({
                'state': target_step['state'], 
                'next_state': next_state, 
                'reward': target_step['reward'], 
                #'reward': reward_q_a_value,
                'action': target_step['action'], 
                'done': target_step['done'], 
                'td_error': td_error
            })

            self.step_deque.popleft()

        # put the new step into the step queue 
        self.step_deque.append({
            'state': state, 
            'reward': reward,
            'action': action,
            'done': done,
            'q_a_value': q_values[action]            
        })

    def get_next_send_step(self):
        if len(self.send_deque) > 0:
            step = self.send_deque[0]
            self.send_deque.popleft()
            return step
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
    Actor explores the environment and accumulates rollout steps sending to the collector.
        - receives model parameters from Learner periodically 
        - sends the steps to the collector
    """

    def __init__(self, actor_index, actor_count, args):
        self.actor_index = actor_index
        self.actor_count = actor_count 
        self.args = args
        self.mq_collector = mq.MqProducer('d5qn_collector', 1000000)
        self.mq_parameter = mq.MqConsumer('d5qn_parameter', 1000000)

    def prepare(self):
        '''
        prepares exploring the env
            - setup atari environment
            - create the NN model 
            - receives the initial NN parameters 
        '''
        msgnum.patch()

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        print(f'device: {device}')
        # setting the number of threads to 1 solves high cpu utilization
        torch.set_num_threads(1)

        self.writer = tensorboardX.SummaryWriter(comment="-{}-actor{}".format(self.args.env, self.actor_index))

        self.env = env_wrappers.make_atari(self.args.env)
        self.env = env_wrappers.wrap_atari_dqn(self.env, self.args)

        self.mq_collector.start()
        self.mq_parameter.start()

        seed = self.args.seed + self.actor_index
        utils.set_global_seeds(seed, use_torch=True)

        self.env.seed(seed)

        # setup learning environment 
        # - check mq 
        # - check learner 
        # - receive initial model parameters 

        self.model = model.DQN(self.env)
        self.model.to(device)

    def explore(self, min_epsilon=0.001):
        """
        explores the environment and forward steps (transitions) to the collector
        """
        div = max(1, self.actor_count - 1)
        self.epsilon = self.args.epsilon_base ** (1 + self.actor_index / div * self.args.epsilon_alpha)
        self.epsilon_decay = self.args.epsilon_decay
        self.storage = ExploreStepStorage(self.args.n_step, 1000, self.args.gamma)
        episode_reward, episode_length, episode_idx, param_age = 0, 0, 0, 0
        state = self.env.reset()
        sum_q_value = 0

        while True:
            action, q_values = self.model.act(torch.FloatTensor(state), self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.storage.add(state, reward, action, done, q_values)

            if self.args.env_render: 
                self.env.render()

            sum_q_value += q_values[action]
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.epsilon = max(0, self.epsilon-self.epsilon_decay)
            self.epsilon = max(0.001, self.epsilon)

            if done or episode_length == self.args.max_episode_length:
                state = self.env.reset()
                self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                print(f"episode: {episode_idx:5.0f}, epsilon: {self.epsilon:1.5f}, reward: {episode_reward:5.1f}, length: {episode_length:5d}, age: {param_age:5d}, q: {sum_q_value/episode_length:4.3f}")
                episode_reward = 0
                episode_length = 0
                episode_idx += 1
                sum_q_value = 0

            # publish step to the collector
            while True: 
                step = self.storage.get_next_send_step()
                if step is not None:
                    m = msgpack.packb(step)
                    self.mq_collector.publish(m)
                else:
                    break

            # get model parameters from Learner
            m = self.mq_parameter.consume()
            if m is not None:
                params = pickle.loads(m)
                self.model.load_state_dict(params)
                param_age += 1

            time.sleep(0.001)

            if self.epsilon <= min_epsilon:
                break;

    def finish(self):
        pass

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
