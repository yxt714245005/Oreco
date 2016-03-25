# -*- encoding: utf-8 -*-

import os
import sys
import time
import functools
import atexit
from threading import Thread
from signal import SIGTERM
from signal import SIGUSR1
from multiprocessing import Process


def thread(is_join=False, num=1):
    def _wrap1(func):
        @functools.wraps(func)
        def _wrap2(*args, **kwargs):
            pros = []
            for x in xrange(num):
                pros.append(Thread(target=lambda: func(*args, **kwargs)))
            for th in pros:
                th.start()
            if is_join:
                for th in pros:
                    th.join()

        return _wrap2

    return _wrap1


def process(is_join=False, num=1):
    def _wrap1(func):
        @functools.wraps(func)
        def _wrap2(*args, **kwargs):
            pros = []
            for _ in xrange(num):
                pros.append(Process(target=lambda: func(*args, **kwargs)))
            for th in pros:
                th.start()
            if is_join:
                for th in pros:
                    th.join()

        return _wrap2

    return _wrap1


class Daemon(object):
    def __init__(self, pidfile, curdir='/', stdin='/dev/null',
                 stdout=sys.stderr, stderr=sys.stderr):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = pidfile
        self.curdir = curdir

    def _daemonize(self):
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as err:
            msg = 'fork #1 failed: %d (%s)\n' % (err.errno, err.strerror)
            sys.stderr.write(msg)
            sys.exit(1)
        os.setsid()
        os.chdir(self.curdir)
        os.umask(0o022)
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as err:
            msg = 'fork #2 failed: %d (%s)\n' % (err.errno, err.strerror)
            sys.stderr.write(msg)
            sys.exit(1)
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(self.stdin, 'r')
        os.dup2(si.fileno(), sys.stdin.fileno())
        atexit.register(self.del_pid)
        pid = str(os.getpid())
        open(self.pidfile, 'w+').write('%s\n' % pid)

    def del_pid(self):
        os.remove(self.pidfile)

    def start(self):
        try:
            pf = open(self.pidfile, 'r')
            pid = int(pf.read().strip())
            pf.close()
        except IOError:
            pid = None

        if pid:
            message = 'pidfile %s already exist. Daemon already running?\n'
            sys.stderr.write(message % self.pidfile)
            sys.exit(1)

        self._daemonize()
        self._run()

    def stop(self):
        try:
            pf = open(self.pidfile, 'r')
            pid = int(pf.read().strip())
            pf.close()
        except IOError:
            pid = None

        if not pid:
            message = 'pidfile %s does not exist. Daemon not running?\n'
            sys.stderr.write(message % self.pidfile)
            return

        try:
            while 1:
                os.kill(pid, SIGUSR1)
                time.sleep(2)
                os.kill(pid, SIGTERM)
        except OSError as err:
            err = str(err)
            if err.find('No such process') > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                sys.exit(1)

    def restart(self):
        self.stop()
        self.start()