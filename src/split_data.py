import os
import argparse
import yaml
import random
import pandas as pd


def get_data(config_file):
    config=read_params(config_file)
    return config

def read_params(config_file):
    with open(config_file) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config


def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['data']['raw_data_processed']
    dest = config['data']['processed_data']
    
    # Create destination directories if they don't exist
    train_dir = os.path.join(dest, '/train')
    test_dir = os.path.join(dest, '/test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Read CSV file containing data
    csv_file = os.path.join(config['data']['raw_data_processed'])
    df = pd.read_csv(csv_file)
    
    # Shuffle data for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data into 80% train and 20% test
    split_idx = int(len(df) * 0.99)
    train_data = df[:split_idx]
    test_data = df[split_idx:]

    # Save split data as CSV files
    try:
        train_data.to_csv(config['processed_data']['train'], index=False)
        test_data.to_csv(config['processed_data']['test'], index=False)
        print("CSV files saved successfully!")
    except Exception as e:
        print(f"Error saving CSV files: {e}")
    
    print(f"Data split completed: {len(train_data)} training samples, {len(test_data)} testing samples")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../params.yaml')
    args = parser.parse_args()
    train_and_test(config_file=args.config)
