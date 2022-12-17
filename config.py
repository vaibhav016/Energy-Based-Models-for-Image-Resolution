# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Union, List

import tensorflow as tf
import yaml


def load_yaml(path):
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)


def preprocess_paths(
        paths: Union[List[str], str], isdir: bool = False
) -> Union[List[str], str]:
    """Expand the path to the root "/" and makedirs

    Args:
        paths (Union[List, str]): A path or list of paths

    Returns:
        Union[List, str]: A processed path or list of paths, return None if it's not path
    """
    if isinstance(paths, list):
        paths = [os.path.abspath(os.path.expanduser(path)) for path in paths]
        for path in paths:
            dirpath = path if isdir else os.path.dirname(path)
            if not tf.io.gfile.exists(dirpath):
                tf.io.gfile.makedirs(dirpath)
        return paths
    if isinstance(paths, str):
        paths = os.path.abspath(os.path.expanduser(paths))
        dirpath = paths if isdir else os.path.dirname(paths)
        if not tf.io.gfile.exists(dirpath):
            tf.io.gfile.makedirs(dirpath)
        return paths
    return None


class Config:
    def __init__(self, data: Union[str, dict]):
        config = data if isinstance(data, dict) else load_yaml(preprocess_paths(data))
        self.ebm_train_config = config.pop("ebm_train_config", {})
        self.ebm_data_config = config.pop("ebm_data_config", {})

        self.gan_train_config = config.pop("gan_train_config", {})
        self.gan_data_config = config.pop("gan_data_config", {})

        for k, v in config.items():
            setattr(self, k, v)
