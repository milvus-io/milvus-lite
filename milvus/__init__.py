"""
An embedded version of Milvus (https://milvus.io/)
"""
import ctypes
import threading
import pathlib
import shutil
import os
import sys
from importlib_resources import files
from distutils.dir_util import copy_tree
import platform

CONFIG_PATH = '/var/bin/e-milvus/configs/'
LIB_PATH = '/var/bin/e-milvus/lib/'
LOG_PATH = '/var/bin/e-milvus/logs/'
EG_PATH = '/var/bin/e-milvus/examples/'
CONFIG_NAME = 'embedded-milvus.yaml'

try:
    shutil.rmtree(LIB_PATH)
except Exception:
    pass

library = None
thr = None

def init():
    # Prepare directories and configuration files
    config = str(files('milvus.configs').joinpath(CONFIG_NAME))
    pathlib.Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(LIB_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(EG_PATH).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CONFIG_PATH + CONFIG_NAME):
        print("# Creating Milvus config for the first time under: " + CONFIG_PATH +
            CONFIG_NAME)
        shutil.copy2(config, CONFIG_PATH)
    copy_tree(pathlib.Path(__file__).parent / 'lib', LIB_PATH)
    copy_tree(pathlib.Path(__file__).parent / 'examples', EG_PATH)

    # Automatically inject variables
    osType = platform.system()
    if osType == 'Linux':
        milvusDL = str(files('milvus.bin').joinpath('embd-milvus.so'))
        linuxLdPreload = os.getenv('LD_PRELOAD')
        if linuxLdPreload == None:
            os.environ['LD_PRELOAD'] = milvusDL
        elif milvusDL not in linuxLdPreload:
            os.environ['LD_PRELOAD'] = ':'.join([linuxLdPreload, milvusDL])
        print("export LD_PRELOAD=" + os.environ['LD_PRELOAD'])

        linuxLdLibraryPath = os.getenv('LD_LIBRARY_PATH')
        if linuxLdLibraryPath == None:
            os.environ['LD_LIBRARY_PATH'] = "/usr/lib:/usr/local/lib:" + LIB_PATH
        elif LIB_PATH not in linuxLdLibraryPath:
            os.environ['LD_LIBRARY_PATH'] = ':'.join([linuxLdLibraryPath, LIB_PATH])
        print("export LD_LIBRARY_PATH=" + os.environ['LD_LIBRARY_PATH'])
    elif osType == 'Darwin':
        macDyld = os.getenv('DYLD_FALLBACK_LIBRARY_PATH')
        if macDyld == None:
            os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = "/usr/lib:/usr/local/lib:" + LIB_PATH
        elif LIB_PATH not in macDyld:
            os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join([macDyld, LIB_PATH])
        print("export DYLD_FALLBACK_LIBRARY_PATH=" + os.environ['DYLD_FALLBACK_LIBRARY_PATH'])

def checkEnvs():
    osType = platform.system()
    if osType == 'Linux':
        milvusDL = str(files('milvus.bin').joinpath('embd-milvus.so'))
        linuxLdPreload = os.getenv('LD_PRELOAD')
        if linuxLdPreload == None or (milvusDL not in linuxLdPreload):
            return False
        linuxLdLibraryPath = os.getenv('LD_LIBRARY_PATH')
        if linuxLdLibraryPath == None or LIB_PATH not in linuxLdLibraryPath:
            return False
    elif osType == 'Darwin':
        macDyld = os.getenv('DYLD_FALLBACK_LIBRARY_PATH')
        if macDyld == None or (LIB_PATH not in macDyld):
            return False
    return True

def before():
    osType = platform.system()
    print('please do the following if you haven not already done so:')
    print('1. install required dependencies: bash ' + LIB_PATH + 'install_deps.sh')
    if osType == 'Linux':
        print('2. export LD_PRELOAD=' + str(files('milvus.bin').joinpath('embd-milvus.so')))
        print('3. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:' + LIB_PATH)
    elif osType == 'Darwin':
        print('2. export DYLD_FALLBACK_LIBRARY_PATH=DYLD_FALLBACK_LIBRARY_PATH:/usr/lib:/usr/local/lib:' + LIB_PATH)


def start():
    global library, thr
    def run_milvus():
        if checkEnvs() == True:
            library.startEmbedded()
        else:
            print("Please execute the following command to complete the environment configuration")
            print('eval $(python -c "import milvus; milvus.init()")')
            sys.exit(1)
    if not (library is None):
        print("Milvus already started")
        return
    library = ctypes.cdll.LoadLibrary(
        files('milvus.bin').joinpath('embd-milvus.so'))
    thr = threading.Thread(target=run_milvus, args=(), kwargs={})
    thr.setDaemon(True)
    thr.start()


def stop():
    osType = platform.system()
    print('to clean up, run:')
    if osType == 'Linux':
        print('export LD_PRELOAD=')
        print('export LD_LIBRARY_PATH=')
    elif osType == 'Darwin':
        print('export DYLD_FALLBACK_LIBRARY_PATH=')
    global library, thr
    try:
        library.stopEmbedded()
        thr.join()
    except Exception:
        print("stop() must be called after start() and shall only be called once.")
        return
