import mq
import msgpack
import time

def test_mq():
    producer = mq.MqProducer('d5qn_actor')
    consumer = mq.MqConsumer('d5qn_actor')

    producer.start()
    consumer.start()

    for i in range(0, 100000):
        producer.publish(msgpack.packb([1, 2, 3]))

    for i in range(0, 100000):
        r = consumer.consume(blocking=True)
        if r is not None:
            m = msgpack.unpackb(r)
            assert m[0] == 1

    producer.close()
    consumer.close()

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

if __name__ == "__main__":
    test_mq()