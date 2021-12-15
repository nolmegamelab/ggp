import pika 
import threading
import queue
import time
import msgpack

class MqProducer(object):
    '''
    MqProducer joins exchange with exchg_name and publish messages to the exchange. 
    The exchange is fanout to broadcast to all consumers joined to that exchange.
    '''

    def __init__(self, exchg_name):
        super(MqProducer, self).__init__()
        self.exchg_name = exchg_name
        self._initMq()

    def start(self):
        self._initMq()
        print(f'mq initialized {self.exchg_name}')
        self.q_channel.start_consuming()

    def publish(self, m):
        assert m is not None
        self.q_channel.basic_publish(
                            exchange=self.exchg_name, 
                            routing_key='', 
                            body=m)

    def close(self):
        self.q_conn.close()

    def join(self):
        pass

    def _initMq(self):
        self.q_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.q_channel = self.q_conn.channel()
        self.q_channel.exchange_declare(self.exchg_name, 'fanout')

    def _on_message(self, ch, method, props, body):
        print(f"received: {body}")
        self.recv_queue.put(body)


class MqConsumer(threading.Thread):
    '''
    MqConsumer joins exchange with exchg_name and consumes messages from the exchange. 
    The exchange is fanout to broadcast to all consumers joined to that exchange.
    '''

    def __init__(self, exchg_name):
        super(MqConsumer, self).__init__()
        self.exchg_name = exchg_name
        self.recv_queue = queue.Queue()
        self.daemon = True
        self._initMq()
        self.message_index = 0

    def consume(self):
        return self.recv_queue.get()

    def close(self):
        try:
            self.q_conn.close()
        except Exception as e:
            print(f'consumer exception in close(): {e}')

    def _initMq(self):
        self.q_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.q_channel = self.q_conn.channel()
        self.q_channel.exchange_declare(self.exchg_name, 'fanout')
        self.q_result = self.q_channel.queue_declare('', exclusive=True)

        self.q_channel.queue_bind(
                            exchange=self.exchg_name, 
                            queue=self.q_result.method.queue)

    def run(self):
        self.q_channel.basic_consume(
                            self.q_result.method.queue,
                            self._on_message)

        print(f'mq_consumer initialized {self.exchg_name}')
        try:
            self.q_channel.start_consuming()
        except Exception as e:
            print(f'consumer exception in run(): {e}')

    def _on_message(self, ch, method, props, body):
        print(f"received: {self.message_index}")
        self.message_index = self.message_index+1
        self.recv_queue.put(body)

