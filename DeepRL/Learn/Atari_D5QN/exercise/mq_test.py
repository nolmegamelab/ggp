import mq_producer
import mq_consumer
import sys
import time
import numpy as np
import msgpack

producer = mq_producer.MqProducer('a', '127.0.0.1')
consumer = mq_consumer.MqConsumer('a')

try:
    producer.start()
    consumer.start()

    # when consumer thread starts later than producer, receive can fail
    time.sleep(1)

    vals = np.arange(1, 10000)
    avals = vals.tolist()
    packed = msgpack.packb(avals)

    for i in range(0, 10000):
        producer.publish(packed)
        consumer.consume()

    time.sleep(1)

    producer.close()
    consumer.close()

    producer.join()
    consumer.join()
except KeyboardInterrupt:
    sys.exit(1)