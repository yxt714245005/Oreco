#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'yangxiantiao'

import os
import sys
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_DIR)
import signal
from utils import Daemon
from config import PID_FILE
from agent import Agent


class AgentDaemon(Daemon):

    def _run(self):
        def sigterm_clean(signum, frame):
            try:
                os.kill(os.getpid(), signal.SIGKILL)
            except:
                pass

        signal.signal(signal.SIGTERM, sigterm_clean)
        agent = Agent()
        agent.main()


if __name__ == '__main__':

    daemon = AgentDaemon(pidfile=PID_FILE)
    if len(sys.argv) == 2:
        if sys.argv[1] in ('start', 'stop', 'restart'):
            {
                'start': daemon.start,
                'stop': daemon.stop,
                'restart': daemon.restart
            }[sys.argv[1]]()
        else:
            print('usage: %s start|stop|restart' % sys.argv[0])
            sys.exit(2)
        sys.exit(0)
    else:
        print('usage: %s start|stop|restart' % sys.argv[0])
        sys.exit(2)
