"""
An embedded version of Milvus (https://milvus.io/)
"""
import ctypes
import threading
import pathlib
import shutil
import os
from importlib_resources import files
from distutils.dir_util import copy_tree

CONFIG_PATH = '/tmp/milvus/configs/'
BIN_PATH = '/tmp/milvus/bin/'
LIB_PATH = '/tmp/milvus/lib/'
LOG_PATH = '/tmp/milvus/logs/'
CONFIG_NAME = 'embedded-milvus.yaml'

try:
    shutil.rmtree(BIN_PATH)
    shutil.rmtree(LIB_PATH)
except Exception:
    pass

config = str(files('milvus.configs').joinpath(CONFIG_NAME))
pathlib.Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(LIB_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

if not os.path.exists(CONFIG_PATH + CONFIG_NAME):
    print("Creating Milvus config for the first time under: " + CONFIG_PATH +
          CONFIG_NAME)
    shutil.copy2(config, CONFIG_PATH)
copy_tree(pathlib.Path(__file__).parent / 'lib', "/tmp/milvus/lib/")
print('---(For linux users) if you are running Milvus for the first time, run milvus.before() for env variables setting instructions---')
print('---otherwise, run milvus.start()---')


def before():
    print('please set the following env variables in bash:\nexport LD_PRELOAD=' + str(files('milvus.bin').joinpath('embd-milvus.so')))
    print('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:/tmp/milvus/lib')


def start():
    library = ctypes.cdll.LoadLibrary(
        files('milvus.bin').joinpath('embd-milvus.so'))

    def run_milvus():
        library.startEmbedded()

    thr = threading.Thread(target=run_milvus, args=(), kwargs={})
    thr.setDaemon(True)
    thr.start()
