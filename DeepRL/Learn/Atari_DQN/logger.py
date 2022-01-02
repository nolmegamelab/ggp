import logging 

class Logger: 

    def __init__(self, file, console=True):
        self.logger = logging.getLogger()

        # 로그의 출력 기준 설정
        self.logger.setLevel(logging.INFO)

        # log 출력 형식
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

        # log 출력
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # log를 파일에 출력
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, m):
        self.logger.debug(m)

    def info(self, m):
        self.logger.info(m)

    def warn(self, m):
        self.logger.warn(m)

    def critical(self, m):
        self.logger.critical(m)

if __name__ == "__main__":
    logger = Logger('test.log')
    logger.info('hello logger')