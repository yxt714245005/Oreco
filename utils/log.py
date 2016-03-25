import os
import logging
from logging.handlers import RotatingFileHandler


class Singleton(object):
    def __new__(type, *args, **kwargs):
        if not '_the_instance' in type.__dict__:
            type._the_instance = object.__new__(type, *args, **kwargs)
        return type._the_instance


class Logger(Singleton):
    _no_handlers = True

    def __init__(self, logfile, loglevel='INFO'):
        self.level = logging._levelNames[loglevel]
        self._setup_logging()
        if self._no_handlers:
            self._setup_handlers(logfile=logfile)

    def _setup_logging(self):
        self.logger = logging.getLogger("recognizer")

    def _setup_handlers(self, logfile):
        if not os.path.exists(logfile):
            logdir=os.path.dirname(logfile)
            if not os.path.exists(logdir):
                os.makedirs(logdir)


        handler = RotatingFileHandler(logfile, maxBytes=100 * 1024 * 1024, backupCount=5)
        self.logger.setLevel(self.level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(module)s.%(funcName)s Line:%(lineno)d %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self._no_handlers = False


if __name__ == "__main__":
    log = Logger('/tmp/test.log').logger
    log.info("log test")