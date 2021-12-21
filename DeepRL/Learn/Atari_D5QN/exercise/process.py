import multiprocessing
import time
import os
import queue

class Producer: 

    def __init__(self, queue): 
        self.process = multiprocessing.Process(target=self.run)
        self.queue = queue

    def start(self): 
        self.process.start()

    def run(self): 
        while True:
            self.queue.put([1, 2, 3])
            time.sleep(0.1)

    def join(self):
        self.process.join()

class Consumer: 

    def __init__(self, queue):
        self.process = multiprocessing.Process(target=self.run)
        self.queue = queue

    def start(self): 
        self.process.start()

    def run(self):
        while True:
            try:
                m = self.queue.get_nowait()
                print(m)
                time.sleep(0.1)
            except queue.Empty:
                print('empty')

    def join(self):
        self.process.join()


def main():
    queue = multiprocessing.Queue(maxsize = 1000000)
    producer = Producer(queue)
    consumer = Consumer(queue)

    #producer.start()
    consumer.start()
    #producer.join()
    consumer.join()

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    multiprocessing.set_start_method("spawn")
    main()