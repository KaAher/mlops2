import numpy as np
import pandas as pd
import yaml
import os
import shutil
import tensorflow as tf
import argparse
from get_data import get_data, read_params

def create_preprocess(config_file):
    """Function to preprocess and organize dataset into train, test, and val directories."""
    
    config = get_data(config_file)  # Load config file
    
    raw_data = config['load_data']['raw_data']  # ✅ Correct key
    full_path = config['load_data']['preprocessed_data']  # ✅ Correct spelling

    # Create main directories
    os.makedirs(os.path.join(full_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(full_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(full_path, 'val'), exist_ok=True)

    classes = ['Accident','Non Accident']

    # Create subdirectories for each class
    for clas in classes:
        os.makedirs(os.path.join(full_path, 'train', clas), exist_ok=True)
        os.makedirs(os.path.join(full_path, 'test', clas), exist_ok=True)
        os.makedirs(os.path.join(full_path, 'val', clas), exist_ok=True)

    # Copy files from raw data to new structure
    for clas in classes:
        src_train = os.path.join(raw_data, 'train', clas)
        src_test = os.path.join(raw_data, 'test', clas)
        src_val = os.path.join(raw_data, 'val', clas)

        dest_train = os.path.join(full_path, 'train', clas)
        dest_test = os.path.join(full_path, 'test', clas)
        dest_val = os.path.join(full_path, 'val', clas)

        # Function to copy files safely
        def copy_files(src, dest):
            if os.path.exists(src) and os.listdir(src):
                print(f"✅ Copying from {src} to {dest}...")
                for f in os.listdir(src):
                    shutil.copy(os.path.join(src, f), os.path.join(dest, f))
            else:
                print(f"⚠ Warning: Directory {src} is missing or empty!")

        # Copy train, test, val data
        copy_files(src_train, dest_train)
        copy_files(src_test, dest_test)
        copy_files(src_val, dest_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    args = parser.parse_args()
    
    create_preprocess(config_file=args.config)
