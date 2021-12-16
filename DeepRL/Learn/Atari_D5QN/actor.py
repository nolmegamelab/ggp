import torch
import tensorboardX 
import numpy as np
import memory
import env_wrappers
import model

class Actor:

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

        self.model = model.DuelingDQN(env)
        self.epsilon = self.args.eps_base ** (1 + self.actor_index / (self.actor_count - 1) * self.args.eps_alpha)
        self.storage = memory.BatchStorage(self.args.n_steps, self.args.gamma)

        param = param_queue.get(block=True)
        self.model.load_state_dict(param)

        print("Received First Parameter!")

    def explore(self):
        outstanding = 0

        episode_reward, episode_length, episode_idx, actor_idx = 0, 0, 0, 0
        state = self.env.reset()

        while True:
            action, q_values = self.model.act(torch.FloatTensor(np.array(state)), self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.storage.add(state, reward, action, done, q_values)

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

            if actor_idx % self.args.update_interval == 0:
                try:
                    param = param_queue.get(block=False)
                    model.load_state_dict(param)
                    print("Updated Parameter..")
                except queue.Empty:
                    pass

            # CHECK 1. 
            # storage가 sliding 하면서 t0_action에 대해 값을 부드럽게 갱신해야 한다. 
            # 현재는 초기화 되기 전의 첫번째 액션에 대해서만 값이 갱신 되는 걸로 보인다. 
            if len(self.storage) == args.send_interval:
                batch, prios = self.storage.make_batch()
                data = pickle.dumps((batch, prios))
                batch, prios = None, None
                self.storage.reset()
                while outstanding >= args.max_outstanding:
                    batch_socket.recv()
                    outstanding -= 1
                batch_socket.send(data, copy=False)
                outstanding += 1
                print("Sending Batch..")


    def finish():
        pass

if __name__ == '__main__':
    actor = Actor(1, 1, {})
    try:
        actor.prepare()
        actor.explore()
    except Exception as e:
        pass
    finally:
        actor.finish()
