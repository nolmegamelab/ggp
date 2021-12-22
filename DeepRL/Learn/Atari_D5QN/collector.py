import multiprocessing
import buffer
import msgpack
import os
import time
import traceback
import gc
import torch
import numpy as np
import msgpack_numpy as m 

import options
import mq

class Collector: 
    '''
    Receives steps from actors and put the steps into PriorityReplayBuffer, then 
    samples them and send those to the Learner
    '''

    def __init__(self, opts): 
        self.opts = opts
        self.finished = False
        self.buffer = buffer.PrioritizedReplayBuffer(opts.replay_buffer_size, opts.alpha) 
        self.mq_collector = mq.MqConsumer('d5qn_collector', 1000000)
        self.mq_learner = mq.MqProducer('d5qn_learner', 1000000)

    def prepare(self):
        m.patch()
        self.mq_collector.start()
        self.mq_learner.start()

    def process(self): 
        loop = 0
        recv = 0
        samples = 0
        while True:
            current_recv_count = 0
            m = self.mq_collector.consume()
            while m is not None:
                step = msgpack.unpackb(m)
                priority = step['td_error'] + self.opts.min_priority
                self.buffer.add(
                    step['state'], 
                    step['action'], 
                    step['reward'], 
                    step['next_state'], 
                    step['done'], 
                    priority)
                recv = recv + 1
                current_recv_count = current_recv_count + 1 
                print(f'recv: {recv}')
                m = self.mq_collector.consume()

            if len(self.buffer) >= self.opts.sample_begin_size: 
                sample_count = max(2, current_recv_count//self.opts.batch_size)
                for i in range(0, sample_count):
                    batch = self.buffer.sample(self.opts.batch_size, self.opts.beta)
                    m = msgpack.packb(batch)
                    self.mq_learner.publish(m)
                    samples += 1
                    print(f'smpl: {samples}')
                    batch = None

            time.sleep(0.001) # to prevent 100% CPU usage
            loop = loop + 1
            if loop % 1000 == 0:
                gc.collect()

    def finish(self):
        self.finished = True

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    multiprocessing.set_start_method('spawn')
    opts = options.Options()
    collector = Collector(opts)
    try:
        collector.prepare()
        collector.process()
        # CTRL+C to stop process
    except Exception as e:
        print(f'exception: {e}')
        traceback.print_exc()
    finally:
        collector.finish()