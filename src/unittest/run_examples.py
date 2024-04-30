import os
import sys
import pathlib
import subprocess


def run_all_examples():
    examples_dir = pathlib.Path(__file__).absolute().parent.parent.parent / 'examples'
    for f in examples_dir.glob('*.py'):
        if str(f).endswith('bfloat16_example.py') or str(f).endswith('dynamic_field.py'):
            continue
        print(str(f))
        p = subprocess.Popen(args=[sys.executable, str(f)])
        p.wait()
        if p.returncode != 0:
            return False
    return True


if __name__ == '__main__':
    if not run_all_examples():
        exit(-1)
    exit(0)


