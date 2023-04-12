from io import StringIO
from typing import Any

import pytest
from milvus import MilvusServerConfig
import yaml


def file_to_yaml(filepath: str) -> Any:
    with open(filepath, encoding='utf-8') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


@pytest.mark.parametrize('text, key, value', (
    (
        """
a:
  b:
    c: 1
  x: hello
    """, 'a.b.c', 2
    ),
    (
        """
x:
  y: 123 ##
a:
  b:
    c: 1 ##
  x: hello
    """, 'a.x', False
    )
))
def test_config_extra_config(tmpdir, text, key, value):
    template_file = tmpdir.join("test.yaml")
    template_file.write(text)
    config = MilvusServerConfig(template=template_file.strpath)
    config.load_template()
    config.base_data_dir = tmpdir.strpath
    config.set(key, value)
    config.write_config()
    milvus_config = tmpdir.join('configs', 'milvus.yaml')
    data = file_to_yaml(milvus_config)
    for w in key.split('.'):
        assert w in data
        data = data[w]
    assert data == value
