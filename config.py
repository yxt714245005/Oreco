# -*- encoding: utf-8 -*-
import os


PROJECT_DIR = os.getcwd()

LOG_FILE = '/var/log/recognizer.log'

PID_FILE = '/var/run/recognizer.pid'

# caffe的python文件夹路径
CAFFE_ROOT = '/dist/dist/caffe_debug/python'

CAFFE_MODEL = 'etc/ocr_iter_500000.caffemodel'
CAFFE_DEPLOY = 'etc/lenet_ocr_yao_s.prototxt'
LABEL_FILE = 'etc/labels_yao_s.txt'

REDIS = {
    'db': 0,
    'host': 'localhost',
    'port': 6379,
    'password': ''
}


REDIS_JOB_KEY = 'recognizer_job'
REDIS_RET_KEY = 'recognizer_ret'

# 预测分类时最大并发进程数
MAX_PREDICT_PROCESS = 10
