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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from torch.utils.data import DataLoader

from main import parse_yaml_config
config = parse_yaml_config('config.yaml')


GLUE_TASKS = ["ax", "cola", "mnli", "mnli_matched", 
"mnli_mismatched", "mrpc", "qnli", "qqp", 
"rte", "sst2", "stsb", "wnli"]

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def preprocess_function(batch):
    if config['dataset']['name'] == 'glue':
        if config['dataset']['task'] is not None:
            sentence1_key, sentence2_key = task_to_keys[config['dataset']['task']]
        else:
            sentence1_key, sentence2_key = task_to_keys["qnli"]
    elif config['dataset']['name'] == 'squad':
        sentence1_key, sentence2_key = "question", "context"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    inputs = tokenizer(batch[sentence1_key], batch[sentence2_key], padding="max_length", truncation=True, max_length=128)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    return batch

def load_data(config):

    print(config)
    if config['dataset']['name'] == 'glue':
        if config['dataset']["task"] != None:
            if config['dataset']["task"] not in GLUE_TASKS:
                raise ValueError("Unknown task", config["task"])
            else:
                data = load_dataset(
                    "glue",
                    config['dataset']["task"],
                    cache_dir=config['cache_dir'],
                )
        else:
            # load sst2 by default
            data = load_dataset(
                "glue",
                'qnli', # can we handle this better?
                cache_dir=config['cache_dir'],
            )
    elif config['dataset']['name'] == 'squad':
        data = load_dataset(
            "squad",
            cache_dir=config['cache_dir'],
        )
    else:
        # TODO: add functionality to load more datasets
        raise ValueError('dataset must be glue or squad')

    return data

    
def process_data(data):    
    dataset_mapped = data.map(preprocess_function, batched=True)
    # dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    
    if config['dataset']['name'] == 'glue':
        train_dataloader = torch.utils.data.DataLoader(dataset_mapped['train'], batch_size=32, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(dataset_mapped['validation'], batch_size=32)
        test_dataloader = torch.utils.data.DataLoader(dataset_mapped['test'], batch_size=32)
    elif config['dataset']['name'] == 'squad':
        train_dataloader = torch.utils.data.DataLoader(dataset_mapped['train'], batch_size=32, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(dataset_mapped['validation'], batch_size=32)
        test_dataloader = None

    # for i, batch in enumerate(train_dataloader):

    #     print(len(batch))
    #     print(i)
    #     print(batch.keys())
    #     print()

    return (train_dataloader, valid_dataloader, test_dataloader)