from importlib_resources import files

println('please run:\nexport LD_PRELOAD=$' + str(files('milvus.bin').joinpath('embd-milvus.so')))
