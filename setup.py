import pathlib
import re
import setuptools
from datetime import datetime

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

from milvus import __version__

setuptools.setup(
    name='milvus',
    author='Milvus Team',
    author_email='milvus-team@zilliz.com',
    description='Embedded Version of Milvus',
    version='2.0.1',
    cmdclass={'bdist_wheel': bdist_wheel},
    url='https://github.com/milvus-io/embd-milvus',
    license='Apache-2.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_dir={'milvus': 'milvus'},
    package_data={
        'milvus': ['bin/*.*', 'configs/*.*', 'lib/*.*'],
    },
    install_requires=['importlib_resources>=5.4.0', 'pymilvus==2.0.1'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6')
