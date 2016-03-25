# -*- encoding: utf-8 -*-
import os
import json
import time
import signal
import traceback
from multiprocessing import (
    Value,
    Queue
)
from config import (
    PROJECT_DIR,
    LOG_FILE,
    CAFFE_MODEL,
    CAFFE_DEPLOY,
    LABEL_FILE,
    REDIS_JOB_KEY,
    MAX_PREDICT_PROCESS
)
from utils import process
from utils.log import Logger
from utils.redis_tools import RedisTool
from specimen.revert import revert_image, is_symbol
from classifier import RecEngine

log = Logger(LOG_FILE).logger


QUEUE_SIZE = 5000
SLEEP_TIME = 0.001


class Agent(object):
    """
    OCR识别引擎
    """

    def __init__(self):
        self._stop = Value('d', 0)
        self.caffe_model = os.path.join(PROJECT_DIR, CAFFE_MODEL)
        self.caffe_deploy = os.path.join(PROJECT_DIR, CAFFE_DEPLOY)
        self.label_file = os.path.join(PROJECT_DIR, LABEL_FILE)

        self.queues = {
            "predict": Queue(maxsize=QUEUE_SIZE),
            "ret_job": Queue(maxsize=QUEUE_SIZE),
        }

        self.labels = self.read_labels(self.label_file)
        self.redis = RedisTool.get_instance()
        self.engine = RecEngine(self.caffe_model,
                                self.caffe_deploy,
                                self.label_file)

    def read_labels(self, label_file):
        """
        Returns a list of strings

        Arguments:
        labels_file -- path to a .txt file
        """
        if not label_file:
            return None

        labels = []
        with open(label_file) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)
        assert len(labels), 'No labels found'
        return labels

    @process(num=2)
    def receive_job(self):
        """
        原始图片的预处理流程
        """
        while 1:
            if self._stop.value:
                log.info("receive_job stopping")
                return

            try:
                # 从redis中获取任务
                job_info = self.redis.rpop(REDIS_JOB_KEY)
                if not job_info:
                    time.sleep(SLEEP_TIME)
                    continue

                job_info = json.loads(job_info)
                log.info('receive queue:{}'.format(job_info))
                jid = job_info.get('jid')
                images = job_info.get('images')
                for index, im in enumerate(images):
                    payload = dict(jid=jid, index=index, image=im)
                    self.queues.get('predict').put(payload)

            except:
                log.error(traceback.format_exc())

    @process(num=MAX_PREDICT_PROCESS)
    def predict(self):
        """
        caffe预测处理流程
        """
        while 1:
            if self._stop.value:
                log.info("predict stopping")
                return

            try:
                pre_queue = self.queues["predict"]

                if pre_queue.empty():
                    time.sleep(SLEEP_TIME)
                    continue

                s = time.time()

                # 根据预处理的图片进行预测
                queue_data = self.queues["predict"].get()
                # log.info('predict queue:{}'.format(str(queue_data)))
                image_array = revert_image(queue_data.get('image'))
                result = self.engine.classify([image_array, ])

                payload = dict(
                    result=result,
                    jid=queue_data.get('jid'),
                    index=queue_data.get('index'),
                    image=queue_data.get('image')
                )
                self.queues.get('ret_job').put(payload)

                log.info(time.time() - s)
            except:
                log.error(traceback.format_exc())

    @process(num=5)
    def ret_job(self):
        """
        记过处理流程
        """
        while 1:
            if self._stop.value:
                log.info("ret_job stopping")
                return

            try:
                pre_queue = self.queues["ret_job"]

                if pre_queue.empty():
                    time.sleep(SLEEP_TIME)
                    continue

                queue_data = self.queues["ret_job"].get()
                log.info('ret queue:{}'.format(str(queue_data)))
                jid = queue_data.get('jid')
                index = queue_data.get('index')
                self.redis.hset(jid, index, json.dumps(queue_data))
            except:
                log.error(traceback.format_exc())

    def main(self):
        def sigterm_stop(signum, frame):
            self._stop.value = 1

        signal.signal(signal.SIGUSR1, sigterm_stop)

        # 启动处理流程
        self.receive_job()
        self.predict()
        self.ret_job()

        while 1:
            time.sleep(10)

if __name__ == '__main__':
    agent = Agent()
    agent.main()
