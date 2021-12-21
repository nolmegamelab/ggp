import mq
import msgpack
import time
import buffer
import numpy as np
import line_profiler
import multiprocessing
import os

def test_mq():
    producer = mq.MqProducer('d5qn_actor', 1000000)
    consumer = mq.MqConsumer('d5qn_actor', 1000000)

    producer.start()
    consumer.start()

    for i in range(0, 100000):
        producer.publish(msgpack.packb([i, 2, 3]))

    for i in range(0, 100000):
        r = consumer.consume(blocking=True)
        if r is not None:
            m = msgpack.unpackb(r)
            print(f'message: {m}')

    producer.close()
    consumer.close()

    producer.join()
    consumer.join()

def test_collector():
    '''
    This function sends a bunch of step data to the collector to test: 
        - 
    '''
    producer = mq.MqProducer('d5qn_collector')

    producer.start()

    for i in range(0, 100000):
        producer.publish(msgpack.packb( { 
            'state' : 1, 
            'action' : 0, 
            'reward' : 1, 
            'next_state': 2, 
            'done' : 0, 
            'td_error' : 1 }))

        time.sleep(0.01)


    producer.close()

def test_buffer_performance(): 
    buf = buffer.PrioritizedReplayBuffer(1000000, 0.6)
    for i in range(0, 100000):
        buf.add([1], 1, 1, [2], 0, np.random.rand() * 10)

    for i in range(0, 10000):
        buf.sample(128, 0.6)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    test_mq()