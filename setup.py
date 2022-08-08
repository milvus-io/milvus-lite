import setuptools

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

setuptools.setup(
    name='milvus',
    author='Milvus Team',
    author_email='milvus-team@zilliz.com',
    description='Embedded Version of Milvus',
    version='2.1.0',
    cmdclass={'bdist_wheel': bdist_wheel},
    url='https://github.com/milvus-io/embd-milvus',
    license='Apache-2.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_dir={'milvus': 'milvus'},
    package_data={
        'milvus': ['bin/*.*', 'configs/*.*', 'lib/*.*', 'examples/*.*'],
    },
    install_requires=['importlib_resources', 'pymilvus==2.1.0'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>3.6')
