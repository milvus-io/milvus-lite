"""Milvus Server
"""

from argparse import ArgumentParser, Action
import logging
import os
import shutil
import signal
import sys
import lzma
from os import makedirs
from os.path import join, abspath, dirname, expandvars, isfile
import re
import subprocess
import socket
from time import sleep
import datetime
from typing import Any, List
import urllib.error
import urllib.request
import hashlib

__version__ = '2.2.16'

LOGGERS = {}


def _initialize_data_files(base_dir) -> None:
    bin_dir = join(base_dir, 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    lzma_dir = join(dirname(abspath(__file__)), 'data', 'bin')
    files = filter(lambda x: x.endswith('.lzma'), os.listdir(lzma_dir))
    files = map(lambda x: x[:-5], files)
    for filename in files:
        orig_file = join(bin_dir, filename)
        lzma_md5_file = orig_file + '.lzma.md5'
        lzma_file = join(lzma_dir, filename) + '.lzma'
        with open(lzma_file, 'rb') as raw:
            md5sum_text = hashlib.md5(raw.read()).hexdigest()
        if isfile(lzma_md5_file):
            with open(lzma_md5_file, 'r', encoding='utf-8') as lzma_md5_fp:
                md5sum_text_pre = lzma_md5_fp.read().strip()
            if md5sum_text == md5sum_text_pre:
                continue
        with lzma.LZMAFile(lzma_file, mode='r') as lzma_fp:
            with open(orig_file, 'wb') as raw:
                raw.write(lzma_fp.read())
                os.chmod(orig_file, 0o755)
            with open(lzma_md5_file, 'w', encoding='utf-8') as lzma_md5_fp:
                lzma_md5_fp.write(md5sum_text)


def _create_logger(usage: str = 'null') -> logging.Logger:
    usage = usage.lower()
    if usage in LOGGERS:
        return LOGGERS[usage]
    logger = logging.Logger(name=f'python_milvus_server_{usage}')
    if usage != 'debug':
        logger.setLevel(logging.FATAL)
    else:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s')
        logger.addHandler(handler)
        handler.setFormatter(formatter)
    LOGGERS[usage] = logger
    return logger


class MilvusServerConfig:
    """ Milvus server config
    """

    RANDOM_PORT_START = 40000

    def __init__(self, **kwargs):
        """create new configuration for milvus server

        Kwargs:
            template(str, optional): template file path

            data_dir(str, optional): base data directory for log and data
        """
        self.base_data_dir = ''
        self.configs: dict = kwargs
        self.logger = _create_logger('debug' if kwargs.get('debug', False) else 'null')

        self.template_file: str = kwargs.get('template', None)
        self.template_text: str = ''
        self.config_key_maps = {}
        self.configurable_items = {}
        self.extra_configs = {}
        self.load_template()
        self.parse_template()
        self.listen_ports = {}

    def update(self, **kwargs):
        """ update configs
        """
        self.configs.update(kwargs)

    def load_template(self):
        """ load config template for milvus server
        """
        if not self.template_file:
            self.template_file = join(dirname(abspath(__file__)), 'data', 'config.yaml.template')
        with open(self.template_file, 'r', encoding='utf-8') as template:
            self.template_text = template.read()

    def parse_template(self):
        """ parse template, lightweight template engine
        for avoid introducing dependencies like: yaml/Jinja2

        We using:
        - {{ foo }} for variable
        - {{ bar: value }} for variable with default values
        - {{ bar(type) }} and {{ bar(type): value }} for type hint
        """
        type_mappings = {
            'int': int,
            'bool': bool,
            'str': str,
            'string': str
        }
        for line in self.template_text.split('\n'):
            matches = re.match(r'.*\{\{(.*)}}.*', line)
            if matches:
                text = matches.group(1)
                original_key = '{{' + text + '}}'
                text = text.strip()
                value_type = str
                if ':' in text:
                    key, val = text.split(':', maxsplit=2)
                    key, val = key.strip(), val.strip()
                else:
                    key, val = text.strip(), None
                if '(' in key:
                    key, type_str = key.split('(')
                    key, type_str = key.strip(), type_str.strip()
                    type_str = type_str.replace(')', '')
                    value_type = type_mappings[type_str]
                self.config_key_maps[original_key] = key
                self.configurable_items[key] = [value_type, self.get_value(val, value_type)]
        self.verbose_configurable_items()

    def verbose_configurable_items(self):
        for key, val in self.configurable_items.items():
            self.logger.debug(
                'Config item %s(%s) with default: %s', key, val[0], val[1])

    def resolve(self):
        self.cleanup_listen_ports()
        self.resolve_all_listen_ports()
        self.resolve_storage()
        for key, value in self.configurable_items.items():
            if value[1] is None:
                raise RuntimeError(f'{key} is still not resolved, please try specify one.')
        # ready to start
        self.cleanup_listen_ports()
        self.write_config()
        self.verbose_configurable_items()

    def resolve_port(self, port_start: int):
        used_ports = self.listen_ports.values()
        used_ports = [x[0] for x in used_ports if len(x) == 2]
        used_ports = set(used_ports)
        for i in range(10000):
            port = port_start + i
            if port not in used_ports:
                sock = self.try_bind_port(port)
                if sock:
                    return port, sock
        return None, None

    def resolve_all_listen_ports(self):
        port_keys = list(filter(lambda x: x.endswith('_port'), self.configurable_items.keys()))
        for port_key in port_keys:
            if port_key in self.configs:
                port = int(self.configs.get(port_key))
                sock = self.try_bind_port(port)
                if not sock:
                    raise RuntimeError(f'set {port_key}={port}, but bind failed')
            else:
                port_start = self.configurable_items[port_key][1]
                port_start = port_start or self.RANDOM_PORT_START
                port_start = int(port_start)
                port, sock = self.resolve_port(port_start)
            self.listen_ports[port_key] = (port, sock)
            self.logger.debug('bind port %d for %s success', port, port_key)
        for port_key, data in self.listen_ports.items():
            self.configurable_items[port_key][1] = data[0]

    def try_bind_port(self, port):
        """ return a socket if bind success, else None
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.listen()
            return sock
        except (OSError, socket.error, socket.gaierror, socket.timeout, ValueError, OverflowError) as ex:
            self.logger.debug('try bind port:%d failed, %s', port, ex)
        return None

    @classmethod
    def get_default_data_dir(cls):
        if sys.platform.lower() == 'win32':
            default_dir = expandvars('%APPDATA%')
            return join(default_dir, 'milvus.io', 'milvus-server', __version__)
        default_dir = expandvars('${HOME}')
        return join(default_dir, '.milvus.io', 'milvus-server', __version__)

    @classmethod
    def get_value_text(cls, val) -> str:
        if isinstance(val, bool):
            return 'true' if val else 'false'
        return str(val)

    @classmethod
    def get_value(cls, text, val_type) -> Any:
        if val_type == bool:
            return text == 'true'
        if val_type == int:
            if not text:
                return 0
            return int(text)
        return text

    def resolve_storage(self):
        self.base_data_dir = self.configs.get('data_dir', self.get_default_data_dir())
        self.base_data_dir = abspath(self.base_data_dir)
        makedirs(self.base_data_dir, exist_ok=True)
        config_dir = join(self.base_data_dir, 'configs')
        logs_dir = join(self.base_data_dir, 'logs')
        storage_dir = join(self.base_data_dir, 'data')
        for subdir in (config_dir, logs_dir, storage_dir):
            makedirs(subdir, exist_ok=True)

        # logs
        if sys.platform.lower() == 'win32':
            self.set('etcd_log_path', 'winfile:///' + join(logs_dir, 'etcd.log').replace('\\', '/'))
        else:
            self.set('etcd_log_path', join(logs_dir, 'etcd.log'))
        self.set('system_log_path', logs_dir)

        # data
        self.set('etcd_data_dir', join(storage_dir, 'etcd.data'))
        self.set('local_storage_dir', storage_dir)
        self.set('rocketmq_data_dir', join(storage_dir, 'rocketmq'))

    def get(self, attr) -> Any:
        return self.configurable_items[attr][1]

    def get_type(self, attr) -> Any:
        return self.configurable_items[attr][0]

    def set(self, attr, val) -> None:
        if attr in self.configurable_items:
            if isinstance(val, self.configurable_items[attr][0]):
                self.configurable_items[attr][1] = val
        else:
            self.extra_configs[attr] = val

    def cleanup_listen_ports(self):
        for data in self.listen_ports.values():
            if data[1]:
                data[1].close()
        self.listen_ports.clear()

    def write_config(self):
        config_file = join(self.base_data_dir, 'configs', 'milvus.yaml')
        os.makedirs(dirname(config_file), exist_ok=True)
        content = self.template_text
        for key, val in self.config_key_maps.items():
            value = self.configurable_items[val][1]
            value_text = self.get_value_text(value)
            content = content.replace(key, value_text)
        content = self.update_extra_configs(content)
        with open(config_file, 'w', encoding='utf-8') as config:
            config.write(content)

    def update_extra_configs(self, content):
        current_key = []
        new_content = ''
        for line in content.splitlines():
            if line.strip().startswith('#'):
                new_content += line + os.linesep
                continue
            matches = re.match(
                r'^( *[a-zA-Z0-9_]+):([^#]*)(#.*)?$', line.rstrip())
            if not matches:
                new_content += line + os.linesep
                continue
            key_with_prefix = matches.group(1).rstrip()
            comment = matches.group(3) or ''
            key = key_with_prefix.strip()
            level = (len(key_with_prefix) - len(key)) // 2
            current_key = current_key[:level]
            current_key.append(key)
            current_key_text = '.'.join(current_key)
            for extra_key, extra_val in self.extra_configs.items():
                if extra_key == current_key_text:
                    if comment.strip():
                        line = f'{key_with_prefix}: {extra_val} #{comment.strip()[1:]}'
                    else:
                        line = f'{key_with_prefix}: {extra_val}'
            new_content += line + os.linesep
        return new_content


class MilvusServer:
    """ Milvus server
    """

    def __init__(self, config: MilvusServerConfig = None, wait_for_started=True, **kwargs):
        """_summary_

        Args:
            config (MilvusServerConfig, optional): the server config.
                Defaults to default_server_config.
            wait_for_started (bool, optional): wait for server started. Defaults to True.

        Kwargs:
        """
        if not config:
            self.config = MilvusServerConfig()
        else:
            self.config = config
        self.config.update(**kwargs)
        self.server_proc = None
        self.proc_fds = {}
        self._debug = kwargs.get('debug', False)
        self.logger = _create_logger('debug' if self._debug else 'null')
        self.webservice_port = 9091
        self.wait_for_started = wait_for_started
        self.show_startup_banner = False

    def get_milvus_executable_path(self):
        """ get where milvus
        """
        if sys.platform.lower() == 'win32':
            join(self.config.base_data_dir, 'bin', 'milvus.exe')
        return join(self.config.base_data_dir, 'bin', 'milvus')

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        self.stop()

    @classmethod
    def prepend_path_to_envs(cls, envs, name, val):
        envs.update({name: ':'.join([val, os.environ.get(name, '')])})

    def cleanup(self):
        if self.running:
            raise RuntimeError('Server is running')
        shutil.rmtree(self.config.base_data_dir, ignore_errors=True)

    def wait(self):
        while self.running:
            sleep(0.1)

    def wait_started(self, timeout=30000):
        """ wait server started

        Args:
            timeout:  timeout in milliseconds, default 30,000

        use http client to visit the health api to check if server ready
        """
        start_time = datetime.datetime.now()
        health_url = f'http://127.0.0.1:{self.webservice_port}/healthz'
        while (datetime.datetime.now() - start_time).total_seconds() < (timeout / 1000) and self.running:
            try:
                with urllib.request.urlopen(health_url, timeout=100) as resp:
                    content = resp.read().decode('utf-8')
                    if 'OK' in content:
                        self.logger.info('Milvus server is started')
                        # still wait 1 seconds to make sure server is ready
                        sleep(1)
                        return
                    else:
                        sleep(0.1)
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError):
                sleep(0.1)
        if self.running:
            raise TimeoutError(f'Milvus not startd in {timeout/1000} seconds')
        else:
            raise RuntimeError('Milvus server already stopped')

    def start(self):
        self.config.resolve()
        _initialize_data_files(self.config.base_data_dir)

        milvus_exe = self.get_milvus_executable_path()
        old_pwd = os.getcwd()
        os.chdir(self.config.base_data_dir)
        envs = os.environ.copy()
        # resolve listen port for METRICS_PORT (restful service), default 9091
        self.webservice_port, sock = self.config.resolve_port(self.webservice_port)
        sock.close()
        envs.update({
            'DEPLOY_MODE': 'STANDALONE',
            'METRICS_PORT': str(self.webservice_port)
        })
        if sys.platform.lower() == 'linux':
            self.prepend_path_to_envs(envs, 'LD_LIBRARY_PATH', dirname(milvus_exe))
        if sys.platform.lower() == 'darwin':
            self.prepend_path_to_envs(envs, 'DYLD_LIBRARY_PATH', dirname(milvus_exe))
        for name in ('stdout', 'stderr'):
            run_log = join(self.config.base_data_dir, 'logs', f'milvus-{name}.log')
            # pylint: disable=consider-using-with
            self.proc_fds[name] = open(run_log, 'w', encoding='utf-8')
        cmds = [milvus_exe, 'run', 'standalone']
        proc_fds = self.proc_fds
        if self._debug:
            self.server_proc = subprocess.Popen(cmds, env=envs)
        else:
            # pylint: disable=consider-using-with
            self.server_proc = subprocess.Popen(cmds, stdout=proc_fds['stdout'], stderr=proc_fds['stderr'], env=envs)
        os.chdir(old_pwd)
        if self.wait_for_started:
            self.wait_started()
        if not self._debug:
            self.show_banner()

    def show_banner(self):
        if self.show_startup_banner:
            print(r"""

    __  _________ _   ____  ______
   /  |/  /  _/ /| | / / / / / __/
  / /|_/ // // /_| |/ / /_/ /\ \
 /_/  /_/___/____/___/\____/___/ {Lite}

 Welcome to use Milvus!
""")
            print(f' Version:   v{__version__}-lite')
            print(f' Process:   {self.server_proc.pid}')
            print(f' Started:   {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            print(f' Config:    {join(self.config.base_data_dir, "configs", "milvus.yaml")}')
            print(f' Logs:      {join(self.config.base_data_dir, "logs")}')
            print('\n Ctrl+C to exit ...')

    def stop(self):
        if self.server_proc:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                self.server_proc.wait(timeout=3)
            self.server_proc = None
        for fd in self.proc_fds.values():
            fd.close()
        self.proc_fds.clear()

    def set_base_dir(self, dir_path):
        self.config.configs.update(data_dir=dir_path)
        self.config.resolve_storage()

    @property
    def running(self) -> bool:
        return self.server_proc is not None

    @property
    def server_address(self) -> str:
        return '127.0.0.1'

    @property
    def config_keys(self) -> List[str]:
        return self.config.configurable_items.keys()

    @property
    def listen_port(self) -> int:
        return int(self.config.get('proxy_port'))

    @listen_port.setter
    def listen_port(self, val: int):
        self.config.set('proxy_port', val)

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, val: bool):
        self._debug = val
        self.logger = _create_logger('debug' if val else 'null')
        self.config.logger = self.logger


default_server = MilvusServer()
debug_server = MilvusServer(MilvusServerConfig(), debug=True)


# pylint: disable=unused-argument
class ExtraConfigAcxtion(Action):
    """ action class for extra config

    the extra config is in format of key=value, the value will be converted to int or float if possible
    for setting a value to subkey, use key.subkey=value
    """
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if '=' not in values:
            raise ValueError(f'Invalid extra config: {values}')
        key, val = values.split('=', 1)
        if val.isdigit():
            val = int(val)
        elif val.replace('.', '', 1).isdigit():
            val = float(val)
        elif val.lower() in ('true', 'false'):
            val = val.lower() == 'true'
        obj = getattr(namespace, self.dest)
        obj[key] = val
        setattr(namespace, self.dest, obj)


def main():
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', dest='debug', default=False, help='enable debug')
    parser.add_argument('--data', dest='data_dir', default='', help='set base data dir for milvus')
    parser.add_argument('--extra-config', dest='extra_config', default={}, help='set extra config for milvus',
                        action=ExtraConfigAcxtion)

    # dynamic configurations
    for key in default_server.config_keys:
        val = default_server.config.get(key)
        if val is not None:
            val_type = default_server.config.get_type(key)
            name = '--' + key.replace('_', '-')
            parser.add_argument(name, type=val_type, default=val, dest=f'x_{key}',
                                help=f'set value for {key} ({val_type.__name__})')

    args = parser.parse_args()

    # select server
    server = debug_server if args.debug else default_server
    server.show_startup_banner = True

    # set base dir if configured
    if args.data_dir:
        server.set_base_dir(args.data_dir)

    # apply configs
    # pylint: disable=protected-access
    for name, value in args._get_kwargs():
        if name.startswith('x_'):
            server.config.set(name[2:], value)
    for key, value in args.extra_config.items():
        server.config.set(key, value)

    signal.signal(signal.SIGINT, lambda sig, h: server.stop())

    try:
        server.start()
    except TimeoutError:
        print('Wait for milvus server started timeout.')
    except RuntimeError:
        print('Milvus server already stopped.')

    server.wait()


if __name__ == '__main__':
    main()
