"""
Module for learner in Ape-X.
"""
import multiprocessing
import os
import pickle
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
import msgpack
import traceback
import msgpack_numpy as m

import buffer
import env_wrappers
import model
import mq
import options
import utils

class TrainStepStorage: 

    def __init__(self, opts): 
        self.mq_learner = mq.MqConsumer('d5qn_learner', 1000000)
        self.mq_parameter = mq.MqProducer('d5qn_parameter', 1000000)
        self.mq_learner.start()
        self.mq_parameter.start()
        self.opts = opts
        self.memory = buffer.PrioritizedReplayBuffer(
                            self.opts.learner_storage_capacity, 
                            self.opts.alpha)

    def pull_batches(self):
        m = self.mq_learner.consume()
        while m is not None:
            batch = msgpack.unpackb(m)

            z = zip(
                batch[0], 
                batch[1], 
                batch[2], 
                batch[3], 
                batch[4], 
                batch[5], 
                batch[6], 
                batch[7])

            for state, action, reward, next_state, done, priority, weight, indice in z :
                self.memory.add(
                    state, 
                    action, 
                    reward, 
                    next_state, 
                    done, 
                    priority)
            
            m = self.mq_learner.consume()
        # end while

    def sample(self):
        b = self.memory.sample(self.opts.batch_size, self.opts.beta)
        return b

    def check_learning_condition(self):
        return len(self.memory) > self.opts.learning_begin_size

    def send_parameters(self, m):
        self.mq_parameter.publish(m)


class Learner: 
    '''
    Learner that receives samples from collector and learns through those samples.
    '''

    def __init__(self, actor_count, opts):
        self.actor_count = actor_count
        self.opts = opts

    def prepare(self):
        m.patch()
        self.env = env_wrappers.make_atari(self.opts.env)
        self.env = env_wrappers.wrap_atari_dqn(self.env, self.opts)

        utils.set_global_seeds(self.opts.seed, use_torch=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.DQN(self.env).to(self.device)
        self.target_model = model.DQN(self.env).to(self.device)

        #self.model.load_state_dict(torch.load("model.pth"))
        self.target_model.load_state_dict(self.model.state_dict())

        self.writer = SummaryWriter(comment="-{}-learner".format(self.opts.env))

        # optimizer = torch.optim.Adam(model.parameters(), opts.lr)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.opts.learning_rate)
                                    #alpha=0.95, 
                                    #eps=1.5e-7, 
                                    #centered=True)

        self.storage = TrainStepStorage(self.opts)

    def train(self):
        learning_loop_count = 0
        ts = time.time()
        sum_loss = 0

        while True:
            self.storage.pull_batches()

            # check alive actors
            # check whether learning can proceed
            if not self._check_learning_conditions():
                time.sleep(0.1)
                continue

            *batch, indices = self.storage.sample()

            loss, priorities = self._forward(batch)
            self._backward(loss)

            sum_loss += loss

            # NOTE: local priority change only (different from the Ape-X paper)
            # local update is not n-step td error. therefore, it is removed 
            # instead, actor model update works in the similar manner (hope)
            # self.storage.update_priorities(indices, priorities)

            batch, indices, priorities = None, None, None
            learning_loop_count += 1

            self.writer.add_scalar("learner/loss", loss, learning_loop_count)

            if learning_loop_count % self.opts.target_update_interval == 0:
                print("Updating Target Network..")
                self.target_model.load_state_dict(self.model.state_dict())

            if learning_loop_count % self.opts.save_interval == 0:
                print("Saving Model..")
                torch.save(self.model.state_dict(), "model.pth")

            if learning_loop_count % self.opts.publish_param_interval == 0:
                params = self.model.state_dict()
                m = pickle.dumps(params)
                self.storage.send_parameters(m)

            if learning_loop_count % self.opts.bps_interval == 0:
                bps = self.opts.bps_interval / (time.time() - ts)
                print("step: {:8}, bps: {:.2f}, loss: {:3.3f}".format(learning_loop_count, bps, sum_loss / self.opts.bps_interval))
                self.writer.add_scalar("learner/BPS", bps, learning_loop_count)
                ts = time.time()
                sum_loss = 0

    def finish(self):
        pass

    def _forward(self, batch):
        states, actions, rewards, next_states, dones, priors, weights = batch

        states_float = np.array(states).astype(np.float32) / 255.0
        next_states_float = np.array(next_states).astype(np.float32) / 255.0
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # convert back to float from uint8 
        states_tensor = torch.FloatTensor(states_float).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states_float).to(self.device)

        q_values = self.model(states_tensor)
        target_next_q_values = self.target_model(next_states_tensor)

        actions_tensor = torch.from_numpy(actions)
        actions_tensor = actions_tensor.type(torch.int64).unsqueeze(1).to(self.device)

        # decrypt following 
        q_a_values = q_values.gather(-1, actions_tensor).squeeze(1)
        next_q_a_values = target_next_q_values.max(-1)[0].unsqueeze(1)

        # rewards가 이전 step의 보상을 포함하고 있어 최종 보상만 감쇄 반영한다. 
        expected_q_a_values = rewards_tensor + self.opts.gamma * next_q_a_values * (1 - dones_tensor)
        expected_q_a_values = expected_q_a_values.to(self.device)

        #  후버로스 계산
        td_error = torch.abs(expected_q_a_values - q_a_values)
        quadratic_part = torch.clip(td_error, 0.0, 1.0)
        linear_part = td_error - quadratic_part

        loss = 0.5 * quadratic_part ** 2 + linear_part
        loss = loss.mean()
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
        #torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.opts.max_norm)
        self.optimizer.step()
        # NOTE: removed total norm calculation

    def _check_learning_conditions(self):
        '''
        Checks whether learning can proceed. 
        '''
        return self.storage.check_learning_condition() 

if __name__ == '__main__': 
    os.environ["OMP_NUM_THREADS"] = "1"
    multiprocessing.set_start_method("spawn")
    opts = options.Options()
    learner = Learner(1, opts)
    try:
        learner.prepare()
        learner.train()
    except Exception as e:
        print(f'exception: {e}')
        traceback.print_exc()
    finally:
        learner.finish()