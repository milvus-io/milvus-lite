import ctypes
import threading
import time
import random
import pathlib
import shutil
import os
from importlib_resources import files
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

CONFIG_PREFIX = '/tmp/milvus/configs/'
CONFIG_NAME = 'embedded-milvus.yaml'

config = str(files('milvus.configs').joinpath(CONFIG_NAME))
pathlib.Path(CONFIG_PREFIX).mkdir(parents=True, exist_ok=True)
if not os.path.exists(CONFIG_PREFIX + CONFIG_NAME):
    print("Creating Milvus config for the first time under:" + CONFIG_PREFIX +
          CONFIG_NAME)
    shutil.copy2(config, CONFIG_PREFIX)

os.environ["LD_PRELOAD"] = str(files('milvus.bin').joinpath('embd-milvus.so'))

library = ctypes.cdll.LoadLibrary(
    files('milvus.bin').joinpath('embd-milvus.so'))


def run_milvus():
    library.startEmbedded()


def is_milvus_alive():
    return thr.is_alive()


def milvus_config_path():
    return CONFIG_PREFIX + CONFIG_NAME


thr = threading.Thread(target=run_milvus, args=(), kwargs={})
thr.setDaemon(True)
thr.start()
