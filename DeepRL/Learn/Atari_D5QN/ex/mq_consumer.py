import pika 
import threading
import queue
import time
import msgpack

class MqConsumer(threading.Thread):

    def __init__(self, exchg_name, host='localhost', port=5672):
        super(MqConsumer, self).__init__()
        self.exchg_name = exchg_name
        self.recv_queue = queue.Queue()
        self.daemon = True
        self.host = host 
        self.port = port
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
            pika.ConnectionParameters(host=self.host, port=self.port)
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


