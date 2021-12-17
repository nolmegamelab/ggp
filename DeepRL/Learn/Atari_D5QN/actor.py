import torch
import tensorboardX 
import numpy as np
import collections 
import env_wrappers
import model
import utils
import options

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
        # when n_step is accumulated, we calculate td_error and prepares to send
        if len(self.step_deque) == self.n_step:
            target_step = self.step_deque[0]
            reward_n_step = self._multi_step_reward(reward)
            next_q_a_max = q_values.max()
            reward_q_a_value = reward_n_step + (self.gamma ** (self.n_step+1)) * next_q_a_max * (1 -done) 
            td_error = reward_q_a_value - target_step['q_a_value']

            # Ape-X paper compute priority while exploring, but it seems to be 
            # more accurate to calculate on the collector using PriorityReplayBuffer 
            # and samples on that priority. The collector, then, can send the batches 
            # to the Learner.
            self.send_deque.append({
                'state': target_step['state'], 
                'reward': target_step['reward'], 
                'action': target_step['action'], 
                'done': target_step['done'], 
                'q_a_value': q_values[action], 
                'reward_n_step': reward_n_step, 
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
        if reward > 0 :
            print(f'reward: {reward}, action: {action}')

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
        pass

    def prepare(self):
        '''
        prepares exploring the env
            - setup atari environment
            - create the NN model 
            - receives the initial NN parameters 
        '''
        self.writer = tensorboardX.SummaryWriter(comment="-{}-actor{}".format(self.args.env, self.actor_index))

        self.env = env_wrappers.make_atari(self.args.env)
        self.env = env_wrappers.wrap_atari_dqn(self.env, self.args)

        seed = self.args.seed + self.actor_index
        utils.set_global_seeds(seed, use_torch=True)

        self.env.seed(seed)

        # setup learning environment 
        # - check mq 
        # - check learner 
        # - receive initial model parameters 

        self.model = model.DuelingDQN(self.env)
        div = max(1, self.actor_count - 1)
        self.epsilon = self.args.eps_base ** (1 + self.actor_index / div * self.args.eps_alpha)
        self.storage = ExploreStepStorage(self.args.n_step, 1000, self.args.gamma)

    def explore(self):
        """
        explores the environment and forward steps (transitions) to the collector
        """
        episode_reward, episode_length, episode_idx, actor_idx = 0, 0, 0, 0
        state = self.env.reset()

        while True:
            action, q_values = self.model.act(torch.FloatTensor(np.array(state)), self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.storage.add(state, reward, action, done, q_values)

            if self.args.render_env: 
                self.env.render()

            state = next_state
            episode_reward += reward
            episode_length += 1
            actor_idx += 1

            if done or episode_length == self.args.max_episode_length:
                state = self.env.reset()
                self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                episode_reward = 0
                episode_length = 0
                episode_idx += 1

            # TODO: get model parameters from Learner

            while True: 
                step = self.storage.get_next_send_step()
                if step is not None:
                    # TODO: send the step
                    pass
                else:
                    break
            

    def finish(self):
        pass

if __name__ == '__main__':
    opts = options.Options()
    actor = Actor(1, 1, opts)
    try:
        actor.prepare()
        actor.explore()
    except Exception as e:
        print(f'exception: {e}')
    finally:
        actor.finish()
