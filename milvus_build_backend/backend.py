import os
import sys
import lzma
import platform
import setuptools.build_meta as _build


def _get_project_dir():
    return os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def _build_milvus_binary():
    project_dir = _get_project_dir()
    status = os.system(f'bash {project_dir}/milvus_binary/build.sh')
    if status != 0:
        raise RuntimeError('Build milvus binary failed')
    # install it to data/bin
    bin_dir = os.path.join(project_dir, 'milvus_binary', 'output')
    to_dir = os.path.join(project_dir, 'src', 'milvus', 'data', 'bin')
    os.makedirs(to_dir, exist_ok=True)
    for file in os.listdir(bin_dir):
        if file.endswith('.txt'):
            continue
        file_from = os.path.join(bin_dir, file)
        file_to = os.path.join(to_dir, f'{file}.lzma')
        with lzma.open(file_to, 'wb') as lzma_file:
            with open(file_from, 'rb') as orig_file:
                print('writeing binary file: ', file)
                lzma_file.write(orig_file.read())


def _get_platform():
    machine_text = platform.machine().lower()
    if sys.platform.lower() == 'darwin':
        if machine_text == 'x86_64':
            return 'macosx_12_0_x86_64'
        elif machine_text == 'arm64':
            return 'macosx_12_0_arm64'
    if sys.platform.lower() == 'linux':
        return f'manylinux2014_{machine_text}'
    if sys.platform.lower() == 'win32':
        return 'win_amd64'


get_requires_for_build_wheel = _build.get_requires_for_build_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _build_milvus_binary()
    name = _build.build_wheel(
        wheel_directory, config_settings, metadata_directory)
    if name.endswith('-none-any.whl'):
        new_name = name.replace('-any.whl', f'-{_get_platform()}.whl')

    os.rename(os.path.join(wheel_directory, name),
              os.path.join(wheel_directory, new_name))
    return new_name
