# -*- encoding: utf-8 -*-
__author__ = 'yangxiantiao'


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            super_obj = super(Singleton, cls)
            cls._instances[cls] = super_obj.__call__(*args, **kwargs)
        return cls._instances[cls]
