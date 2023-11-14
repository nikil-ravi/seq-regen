import argparse
import yaml

from data import load_data
from generation import *

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

    # do something with dataset
    features = get_text_features("hello world", config) # this is a placeholder
    print(features)

if __name__ == '__main__':
    main()
