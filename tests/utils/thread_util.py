import logging
import threading
from typing import Callable


class ConcurrencyObj:
    def __init__(self, func_obj: Callable, interval=0, stop_flag=False):
        self.func_obj = func_obj
        self.interval = interval
        self.stop_flag = stop_flag

    def set_stop_flag(self, flag=True):
        self.stop_flag = flag

    def thread_func(self, *args, **kwargs):
        self.func_obj(*args, **kwargs)
        if self.stop_flag is False:
            t = threading.Timer(interval=self.interval, function=self.thread_func, args=args, kwargs=kwargs)
            t.start()

    def multi_func(self, num, *args, **kwargs):
        logging.info(f"multi_func start ")
        for i in range(num):
            logging.info("thread number: {}".format(i))
            self.thread_func(*args, **kwargs)

