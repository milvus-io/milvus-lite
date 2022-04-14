import ctypes
import threading
import time
import random
import pathlib
import shutil
import os
from importlib_resources import files
from distutils.dir_util import copy_tree
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

CONFIG_PATH = '/tmp/milvus/configs/'
LOG_PATH = '/tmp/milvus/logs/'
CONFIG_NAME = 'embedded-milvus.yaml'

config = str(files('milvus.configs').joinpath(CONFIG_NAME))
pathlib.Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
if not os.path.exists(CONFIG_PATH + CONFIG_NAME):
    print("Creating Milvus config for the first time under:" + CONFIG_PATH +
          CONFIG_NAME)
    shutil.copy2(config, CONFIG_PATH)
shutil.copy2(str(files('milvus.bin').joinpath('embd-milvus.so')), '/tmp/milvus/lib/')
copy_tree(pathlib.Path(__file__).parent / 'lib', "/tmp/milvus/lib/")

library = ctypes.cdll.LoadLibrary(
    files('milvus.bin').joinpath('embd-milvus.so'))


def run_milvus():
    library.startEmbedded()


def is_milvus_alive():
    return thr.is_alive()


def milvus_config_path():
    return CONFIG_PATH + CONFIG_NAME


thr = threading.Thread(target=run_milvus, args=(), kwargs={})
thr.setDaemon(True)
thr.start()
