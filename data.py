#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
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

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

def load_data(config):
    if config['dataset'] == 'glue':
        data = load_dataset(
            "glue",
            config['task_name'] if 'task_name' in config else 'sst2', #handle this better
            cache_dir=config['cache_dir'],
        )
    elif config['dataset'] == 'squad':
        data = load_dataset(
            "squad",
            cache_dir=config['cache_dir'],
        )
    else:
        # TODO: add functionality to load more datasets
        raise ValueError('dataset must be glue or squad')

    return data

    