"""
Module for learner in Ape-X.
"""
import time
import torch
from tensorboardX import SummaryWriter
import msgpack
import traceback

import buffer
import env_wrappers
import model
import mq
import options
import utils

class TrainStepStorage: 

    def __init__(self, opts): 
        self.mq_learner = mq.MqConsumer('d5qn_learner')
        self.mq_parameter = mq.MqProducer('d5qn_parameter')
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

            for state, action, reward, next_state, done, weight, indice, priority in z :
                self.memory.add(
                    torch.tensor(state), 
                    action, 
                    reward, 
                    torch.tensor(next_state), 
                    done, 
                    priority)
            
            m = self.mq_learner.consume()
        # end while

    def sample(self):
        b = self.memory.sample(self.opts.batch_size, self.opts.beta)
        return b

    def update_priorities(self, indices, priorities):
        self.memory.update_priorities(indices, priorities)

    def check_learning_condition(self):
        return len(self.memory) > self.opts.learning_begin_size


class Learner: 
    '''
    Learner that receives samples from collector and learns through those samples.
    '''

    def __init__(self, actor_count, opts):
        self.actor_count = actor_count
        self.opts = opts

    def prepare(self):
        self.env = env_wrappers.make_atari(self.opts.env)
        self.env = env_wrappers.wrap_atari_dqn(self.env, self.opts)

        utils.set_global_seeds(self.opts.seed, use_torch=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.DuelingDQN(self.env).to(self.device)
        self.target_model = model.DuelingDQN(self.env).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.writer = SummaryWriter(comment="-{}-learner".format(self.opts.env))

        # optimizer = torch.optim.Adam(model.parameters(), opts.lr)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), 
                                    self.opts.learning_rate,
                                    alpha=0.95, 
                                    eps=1.5e-7, 
                                    centered=True)

        self.storage = TrainStepStorage(self.opts)

    def train(self):
        learning_loop_count = 0
        ts = time.time()

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

            # NOTE: local priority change only (different from the Ape-X paper)
            self.storage.update_priorities((indices, priorities))

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
                # send paramters to actors
                pass
                #param_queue.put(self.model.state_dict())

            if learning_loop_count % self.opts.bps_interval == 0:
                bps = self.opts.bps_interval / (time.time() - ts)
                print("Step: {:8} / BPS: {:.2f}".format(learning_loop_count, bps))
                self.writer.add_scalar("learner/BPS", bps, learning_loop_count)
                ts = time.time()

    def finish(self):
        pass

    def _forward(self, batch):
        states, actions, rewards, next_states, dones, prorities, weights = batch

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        target_next_q_values = self.target_model(next_states)

        q_a_values = q_values.gather(-1, actions.unsqueeze(1)).squeeze(1)
        next_actions = next_q_values.max(-1)[1].unsqueeze(1)
        next_q_a_values = target_next_q_values.gather(-1, next_actions).squeeze(1)

        # rewards가 이전 step의 보상을 포함하고 있어 최종 보상만 감쇄 반영한다. 
        expected_q_a_values = rewards + (self.opts.gamma ** self.opts.n_steps) * next_q_a_values * (1 - dones)

        td_error = torch.abs(expected_q_a_values.detach() - q_a_values)
        priorities = (td_error + -1e-6).data.cpu().numpy()

        loss = torch.where(td_error < -1, 0.5 * td_error ** 2, td_error - 0.5)
        loss = (loss * weights).mean()
        return loss, priorities


    def _backward(self, loss):
        """
        Update parameters with loss
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.opts.max_norm)
        self.optimizer.step()
        # NOTE: removed total norm calculation

    def _check_learning_conditions(self):
        '''
        Checks whether learning can proceed. 
        '''
        return self.storage.check_learning_condition() 

if __name__ == '__main__': 
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