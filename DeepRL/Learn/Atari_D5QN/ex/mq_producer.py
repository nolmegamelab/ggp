import pika 
import threading
import queue
import time
import msgpack

class MqProducer(object):

    def __init__(self, exchg_name, host='localhost', port=5672):
        super(MqProducer, self).__init__()
        self.exchg_name = exchg_name
        self.host=host
        self.port=port
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
            pika.ConnectionParameters(host=self.host, port=self.port)
        )
        self.q_channel = self.q_conn.channel()
        self.q_channel.exchange_declare(self.exchg_name, 'fanout')

    def _on_message(self, ch, method, props, body):
        print(f"received: {body}")
        self.recv_queue.put(body)

