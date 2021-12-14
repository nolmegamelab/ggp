import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare('hello')

def callback(ch, method, properties, body):
    print(" [x] Received %r" % body)

channel.basic_consume('hello', callback, True)

channel.start_consuming()