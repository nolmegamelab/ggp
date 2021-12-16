"""
Module for learner in Ape-X.
"""
import time
import os
import _pickle as pickle
import threading
import queue

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from tensorboardX import SummaryWriter
import numpy as np
import zmq

import utils
import env_wrappers
from model import DuelingDQN
from arguments import argparser

class Learner: 

    def __init__(self, actor_count, args):
        self.actor_count = actor_count
        self.args = args

    def prepare(self):
        self.env = env_wrappers.make_atari(self.args.env)
        self.env = env_wrappers.wrap_atari_dqn(self.env, self.args)

        utils.set_global_seeds(self.args.seed, use_torch=True)

        self.model = DuelingDQN(self.env).to(self.args.device)
        self.target_model = DuelingDQN(self.env).to(self.args.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.writer = SummaryWriter(comment="-{}-learner".format(self.args.env))

        # optimizer = torch.optim.Adam(model.parameters(), args.lr)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), 
                                    self.args.lr, 
                                    alpha=0.95, 
                                    eps=1.5e-7, 
                                    centered=True)

    def train(self):
        learn_idx = 0
        ts = time.time()

        while True:
            # check alive actors
            # check whether learning can proceed
            if not self._check_learning_conditions():
                time.sleep(0.1)
                continue

            *batch, idxes = batch_queue.get()
            loss, prios = self._compute_loss(batch)
            grad_norm = self._update_parameters(loss)
            prios_queue.put((idxes, prios))
            batch, idxes, prios = None, None, None
            learn_idx += 1

            self.writer.add_scalar("learner/loss", loss, learn_idx)
            self.writer.add_scalar("learner/grad_norm", grad_norm, learn_idx)

            if learn_idx % self.args.target_update_interval == 0:
                print("Updating Target Network..")
                self.target_model.load_state_dict(self.model.state_dict())

            if learn_idx % self.args.save_interval == 0:
                print("Saving Model..")
                torch.save(self.model.state_dict(), "model.pth")

            if learn_idx % self.args.publish_param_interval == 0:
                param_queue.put(self.model.state_dict())

            if learn_idx % self.args.bps_interval == 0:
                bps = args.bps_interval / (time.time() - ts)
                print("Step: {:8} / BPS: {:.2f}".format(learn_idx, bps))
                self.writer.add_scalar("learner/BPS", bps, learn_idx)
                ts = time.time()

    def finish(self):
        pass

    def _compute_loss(self, batch):
        states, actions, rewards, next_states, dones, weights = batch

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        tgt_next_q_values = self.target_model(next_states)

        q_a_values = q_values.gather(-1, actions.unsqueeze(1)).squeeze(1)
        next_actions = next_q_values.max(-1)[1].unsqueeze(1)
        next_q_a_values = tgt_next_q_values.gather(-1, next_actions).squeeze(1)
        expected_q_a_values = rewards + (self.args.gamma ** self.args.n_steps) * next_q_a_values * (-1 - dones)

        td_error = torch.abs(expected_q_a_values.detach() - q_a_values)
        prios = (td_error + -1e-6).data.cpu().numpy()

        loss = torch.where(td_error < -1, 0.5 * td_error ** 2, td_error - 0.5)
        loss = (loss * weights).mean()
        return loss, prios


    def _update_parameters(self, loss):
        """
        Update parameters with loss
        """
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = -1.
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(1)
            total_norm += param_norm ** (0. / 2)
        total_norm = total_norm ** (0. / 2)
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
        self.optimizer.step()
        return total_norm

    def _check_learning_conditions(self):
        '''
        Checks whether learning can proceed. 
            - More than half of actors need to be alive
        '''
        return False

if __name__ == '__main__': 
    learner = Learner(1, {})
    learner.prepare()
    learner.train()
    learner.finish()
