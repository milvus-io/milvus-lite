# Copyright (C) 2019-2024 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import sys
import os
import pathlib
import unittest
from typing import List
import subprocess
import platform

from setuptools import setup, find_namespace_packages, Extension
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import shutil


MILVUS_BIN = 'milvus'


class CMakeBuild(_bdist_wheel):
    def finalize_options(self):
        if sys.platform.lower() == 'linux':
            self.plat_name = f"manylinux2014_{platform.machine().lower()}"
        elif sys.platform.lower() == 'darwin':
            self.plat_name = f"macosx_{platform.machine().lower()}"
        return super().finalize_options()

    def copy_lib(self, lib_path, dst_dir, pick_libs):
        name = pathlib.Path(lib_path).name
        new_file = os.path.join(dst_dir, name)
        for lib_prefix in pick_libs:
            if name.startswith(lib_prefix):
                shutil.copy(lib_path, new_file)
                continue

    def _pack_macos(self, src_dir: str, dst_dir: str):
        mac_pkg = ['libknowhere', 'libmilvus',
                   'libgflags_nothreads', 'libglog',
                   'libtbb', 'libgomp',
                   'libdouble-conversion']
        milvus_bin = pathlib.Path(src_dir) / MILVUS_BIN
        out_str = subprocess.check_output(['otool', '-L', str(milvus_bin)])
        lines = out_str.decode('utf-8').split('\n')
        all_files = []
        for line in lines[1:]:
            r = line.split(' ')
            if not r[0].endswith('dylib'):
                continue
            self.copy_lib(r[1].strip().split(' ')[0].strip(), dst_dir, mac_pkg)

    def _pack_linux(self, src_dir: str, dst_dir: str):
        linux_pkg = ['libknowhere', 'libmilvus',
                     'libgflags_nothreads', 'libglog',
                     'libtbb', 'libm', 'libgcc_s',
                     'libgomp', 'libopenblas',
                     'libdouble-conversion', 'libz',
                     'libgfortran', 'libquadmath']
        milvus_bin = pathlib.Path(src_dir) / MILVUS_BIN
        out_str = subprocess.check_output(['ldd', str(milvus_bin)])
        lines = out_str.decode('utf-8').split('\n')
        all_files = []
        for line in lines:
            r = line.split("=>")
            if len(r) != 2:
                continue
            self.copy_lib(r[1].strip().split(' ')[0].strip(), dst_dir, linux_pkg)

    def run(self):
        build_lib = self.bdist_dir
        build_temp = os.path.abspath(os.path.join(os.path.dirname(build_lib), 'build_milvus'))

        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        #clean build temp
        shutil.rmtree(os.path.join(build_temp, 'lib'), ignore_errors=True)
        extdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ
        env['LD_LIBRARY_PATH'] = os.path.join(build_temp, 'lib')
        subprocess.check_call(['conan', 'install', extdir, '--build=missing', '-s', 'build_type=Release'], cwd=build_temp, env=env)
        subprocess.check_call(['cmake', extdir, '-DENABLE_UNIT_TESTS=OFF'], cwd=build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--', '-j48'],
                              cwd=build_temp,
                              env=env,
                              )
        dst_lib_path = os.path.join(build_lib, 'milvus_lite/lib')
        shutil.rmtree(dst_lib_path, ignore_errors=True)
        os.makedirs(dst_lib_path)
        shutil.copy(os.path.join(build_temp, 'lib', MILVUS_BIN), os.path.join(dst_lib_path, MILVUS_BIN))
        if sys.platform.lower() == 'linux':
            self._pack_linux(os.path.join(build_temp, 'lib'), dst_lib_path)
        elif sys.platform.lower() == 'darwin':
            self._pack_macos(os.path.join(build_temp, 'lib'), dst_lib_path)
        else:
            raise RuntimeError('Unsupport platform: %s', sys.platform)
        
        super().run()


def test_suite():
    test_loader = unittest.TestLoader()
    tests = test_loader.discover('tests', pattern='test_*.py')
    return tests


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name, encoding='utf-8') as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]

setup(name='milvus-lite',
      version='2.4.2',
      description='A lightweight version of Milvus wrapped with Python.',
      author='Milvus Team',
      author_email='milvus-team@zilliz.com',
      url='https://github.com/milvus-io/milvus-lite.git',
      test_suite='setup.test_suite',
      package_dir={'': 'src'},
      packages=find_namespace_packages('src'),
      package_data={},
      include_package_data=True,
      python_requires='>=3.7',
      cmdclass={"bdist_wheel": CMakeBuild},
      long_description=open("../README.md", "r", encoding="utf-8").read(),
      long_description_content_type='text/markdown'
      )
