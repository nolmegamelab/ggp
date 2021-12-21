import multiprocessing
from numpy.core.arrayprint import DatetimeFormat
import pika 
import queue
import time

class MqProducer:
    '''
    MqProducer joins exchange with exchg_name and publish messages to the exchange. 
    The exchange is fanout to broadcast to all consumers joined to that exchange.

        - self.daemon makes this thread exit when the main thread exits
    '''

    def __init__(self, exchg_name, size):
        self.exchg_name = exchg_name
        self.closed = False

    def start(self):
        self._initMq()

    def publish(self, m):
        self._publish(m)

    def close(self):
        '''
        close the channel and stops thread
        '''
        self.closed = True
        self.q_conn.close()

    def join(self): 
        self.process.join()

    def _initMq(self):
        self.q_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.q_channel = self.q_conn.channel()
        self.q_channel.exchange_declare(self.exchg_name, 'fanout')

        print(f'MqProducer initialized {self.exchg_name}')

    def _publish(self, m):
        # TODO: get a serializer as a function and use it to serialize the payload to body
        if m is not None:
            try:
                self.q_channel.basic_publish(
                            exchange=self.exchg_name, 
                            routing_key='', 
                            body=m)
            except Exception as e:
                print(f'producer exception in _publish: {e}')


class MqConsumer:
    '''
    MqConsumer joins exchange with exchg_name and consumes messages from the exchange. 
    The exchange is fanout to broadcast to all consumers joined to that exchange.

        - self.daemon makes this thread exit when the main thread exits
    '''

    def __init__(self, exchg_name, size):
        self.exchg_name = exchg_name
        self.process = multiprocessing.Process(target=self.run, daemon=True)
        self.queue = multiprocessing.Queue(maxsize=size)
        self.closed = False
        self.message_index = 0

    def consume(self, blocking=False):
        if not blocking:
            if self.queue.empty():
                return None
        return self.queue.get()

    def start(self): 
        self.process.start()

    def close(self):
        self.closed = True
        try:
            pass
            #different interpreter creates q_conn
            #self.q_conn.close()
        except Exception as e:
            print(f'MqConsumer exception in close(): {e}')

    def join(self): 
        self.process.join()

    def _initMq(self):
        self.q_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.q_channel = self.q_conn.channel()
        self.q_channel.exchange_declare(self.exchg_name, 'fanout')
        self.q_result = self.q_channel.queue_declare('', exclusive=True, auto_delete=True)

        self.q_channel.queue_bind(
                            exchange=self.exchg_name, 
                            queue=self.q_result.method.queue)

    def run(self):
        self._initMq()
        self.q_channel.basic_consume(
                            self.q_result.method.queue,
                            self._on_message, 
                            auto_ack=True)

        print(f'MqConsumer initialized {self.exchg_name}')
        try:
            self.q_channel.start_consuming()
        except Exception as e:
            print(f'consumer exception in run(): {e}')

    def _on_message(self, ch, method, props, body):
        # TODO: logging
        #print(f"received: {self.message_index}")
        self.message_index = self.message_index+1
        self.queue.put(body)
        if self.closed: 
            self.q_channel.stop_consuming()

