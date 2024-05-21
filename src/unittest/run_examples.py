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
import pathlib
import subprocess


def run_all(py_path):
    for f in py_path.glob('*.py'):
        if str(f).endswith('bfloat16_example.py') or str(f).endswith('dynamic_field.py'):
            continue
        print(str(f))
        p = subprocess.Popen(args=[sys.executable, str(f)])
        p.wait()
        if p.returncode != 0:
            return False
    return True


if __name__ == '__main__':
    examples_dir = pathlib.Path(__file__).absolute().parent.parent.parent / 'examples'
    if not run_all(examples_dir):
        exit(-1)
    pytest = pathlib.Path(__file__).absolute().parent.parent.parent / 'tests'
    if not run_all(pytest):
        exit(-1)    
    exit(0)


