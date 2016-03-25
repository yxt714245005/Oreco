# -*- encoding: utf-8 -*-

import json
import time
import traceback
import datetime
import sys
import redis

REDIS = {
    'db': 0,
    'host': 'localhost',
    'port': 6379,
    'password': ''
}

REDIS_JOB_KEY = 'recognizer_job'
REDIS_RET_KEY = 'recognizer_ret'


class RedisTool(object):
    """
    redis接口封装工具
    """
    redis_db = REDIS.get('db')
    redis_host = REDIS.get('host')
    redis_port = REDIS.get('port')
    redis_password = REDIS.get('password')
    pool = redis.ConnectionPool(db=redis_db,
                                host=redis_host,
                                port=int(redis_port),
                                password=redis_password)

    def __init__(self, db=''):
        self.redis = redis.Redis(connection_pool=self.pool)
        self.db = db

    @classmethod
    def get_instance(cls):
        return redis.Redis(connection_pool=cls.pool)

    def hgetall(self, key):
        try:
            self.redis.hgetall(key)
        except:
            return None


def gen_jid():
    """
    生成一个jid
    """
    return "{0:%Y%m%d%H%M%S%f}".format(datetime.datetime.now())


def get_text(images):
    """
    把图片列表加入到任务处理队列
    @params images list: 图片绝对路径列表
    return 任务处理id
    """

    if not images:
        return None

    redis_obj = RedisTool.get_instance()
    jid = gen_jid()

    payload = dict(
        jid=jid,
        images=images
    )
    redis_obj.lpush(REDIS_JOB_KEY, json.dumps(payload))

    while 1:
        try:
            ret = redis_obj.hgetall(jid)
            if ret and len(ret.keys()) == len(images):
                sorted_ret = sorted(ret.items(), key=lambda x: int(x[0]))
                ret = [json.loads(d[1]).get('result')[0] for d in sorted_ret]
                return dict(image=images, result=ret)
        except:
            print traceback.format_exc()

        time.sleep(0.001)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python recognizer.py image_path1 image_path2 ...'
        exit(0)

    images = sys.argv[1:]
    # images = ['/data/test/mkimg/backup/en62/train/0/0_1.jpg', ]
    start = time.time()
    result = get_text(images)
    for i, item in enumerate(result.get('result')):
        print i, images[i]
        print '=' * 30
        for cl in item:
            print cl[0], ' ==> ', cl[1]

        print
    end = time.time()
    print "use times {} s.".format(end-start)