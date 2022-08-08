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

CONFIG_PATH = '/var/bin/e-milvus/configs/'
LIB_PATH = '/var/bin/e-milvus/lib/'
LOG_PATH = '/var/bin/e-milvus/logs/'
EG_PATH = '/var/bin/e-milvus/examples/'
CONFIG_NAME = 'embedded-milvus.yaml'

try:
    shutil.rmtree(LIB_PATH)
except Exception:
    pass

config = str(files('milvus.configs').joinpath(CONFIG_NAME))
pathlib.Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(LIB_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(EG_PATH).mkdir(parents=True, exist_ok=True)

if not os.path.exists(CONFIG_PATH + CONFIG_NAME):
    print("creating Milvus config for the first time under: " + CONFIG_PATH +
          CONFIG_NAME)
    shutil.copy2(config, CONFIG_PATH)
copy_tree(pathlib.Path(__file__).parent / 'lib', LIB_PATH)
copy_tree(pathlib.Path(__file__).parent / 'examples', EG_PATH)
print('--- if you are running Milvus for the first time, type milvus.before() for pre-run instructions ---')
print('--- otherwise, type milvus.start() ---')

library = None
thr = None


def before():
    print('please do the following if you haven not already done so:')
    print('1. install required dependencies: bash ' + LIB_PATH + 'install_deps.sh')
    print('2. export LD_PRELOAD=' + str(files('milvus.bin').joinpath('embd-milvus.so')))
    print('3. (on Linux systems) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:' + LIB_PATH)
    print('   (on MacOS systems) export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/lib:/usr/local/lib:' + LIB_PATH)


def start():
    global library, thr
    def run_milvus():
        library.startEmbedded()
    if not (library is None):
        print("Milvus already started")
        return
    library = ctypes.cdll.LoadLibrary(
        files('milvus.bin').joinpath('embd-milvus.so'))
    thr = threading.Thread(target=run_milvus, args=(), kwargs={})
    thr.setDaemon(True)
    thr.start()


def stop():
    print('to clean up, run:')
    print('export LD_PRELOAD=')
    print('(on Linux) export LD_LIBRARY_PATH=')
    print('(on MacOS) export DYLD_LIBRARY_PATH=')
    global library, thr
    try:
        library.stopEmbedded()
        thr.join()
    except Exception:
        print("stop() must be called after start() and shall only be called once.")
        return
