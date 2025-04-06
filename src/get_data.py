import yaml

def get_data(config_file):
    config=read_params(config_file)
    return config

def read_params(config_file):
    with open(config_file) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config