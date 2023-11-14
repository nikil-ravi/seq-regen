import argparse
import yaml

from generation import *
import torch
from generation import *
from data import *

# read local config file args
def parse_yaml_config(file_path):
    """
    Parses a YAML configuration file.

    Args:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The parsed YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)

def main():
    config = parse_yaml_config('config.yaml')
    print(config)
    dataset = load_data(config)
    
    # these are datasets, convert them to dataloaders after this
    if config['dataset']['name'] == 'glue':
        train, valid, test = dataset['train'], dataset['validation'], dataset['test']
    elif config['dataset']['name'] == 'squad':
        train, valid = dataset['train'], dataset['validation']

    train_loader, valid_loader, test_loader = process_data(dataset) # note: test_loader does not exist atm for squad

    # can get features
    features = get_text_features("hello world", config) # this is a placeholder

    # can unmask
    unmasked = unmask("welcome [MASK] the awesome world of generative modeling") # this is a placeholder
    print(unmasked)
    #print(train)

    # iterate through train data
    for i in range(len(train[:10])):
        print(train[i])

    # do something with the data loaders in batches....
    # for i, batch in enumerate(train_dataloader):

    #     print(len(batch))
    #     print(i)
    #     print(batch.keys())
    #     print()

if __name__ == '__main__':
    main()
