# -*- encoding: utf-8 -*-

import redis
from config import *


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
