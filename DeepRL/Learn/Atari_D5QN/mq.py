import pika 
import threading
import queue
import time
import msgpack

class MqProducer(threading.Thread):
    '''
    MqProducer joins exchange with exchg_name and publish messages to the exchange. 
    The exchange is fanout to broadcast to all consumers joined to that exchange.
    '''

    def __init__(self, exchg_name):
        super(MqProducer, self).__init__()
        self.exchg_name = exchg_name
        self.queue = queue.Queue()
        self._initMq()

    def publish(self, m):
        self.queue.put(m)

    def close(self):
        self.q_conn.close()

    def run(self):
        while True:
            self._publish()

    def _initMq(self):
        self.q_conn = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.q_channel = self.q_conn.channel()
        self.q_channel.exchange_declare(self.exchg_name, 'fanout')

    def _publish(self):
        m = self.queue.get()
        # TODO: get a serializer as a function and use it to serialize the payload to body
        if m is not None:
            try:
                self.q_channel.basic_publish(
                            exchange=self.exchg_name, 
                            routing_key='', 
                            body=m)
            except Exception as e:
                print(f'producer exception in _publish: {e}')
        else:
            time.sleep(0.001)


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

