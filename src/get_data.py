import numpy as np
import pandas as pd
import yaml
import os
import shutil
import tensorflow as tf
import argparse
from distutils.command.config import config

def get_data(config_file):
    config=read_params(config_file)
    return config

def read_params(config_file):
    with open(config_file) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

if __name__ == '__main__':

    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    
    a=get_data(config_file=passed_args.config)
