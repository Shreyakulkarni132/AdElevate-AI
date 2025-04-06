import pickle
import pandas as pd
import numpy as np
from get_data import get_data, read_params
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../params.yaml')
    args = parser.parse_args()
    config_file=args.config
    config = get_data(config_file)

    with open(config['models']['multiple_regression'] ,"rb") as file:
        model = pickle.load(file)

    # input data
    new_data = pd.DataFrame({
        'admiration':[2.2],
        'amusement':[0.33],
        'anger':[0.28],
        'annoyance':[1.05],
        'approval':[24.57],
        'caring':[1.69],
        'confusion':[0.72],
        'curiosity':[0.75],
        'desire':[0.62],
        'disappointment':[0.38],
        'disapproval':[0.77],
        'disgust':[0.17],
        'embarrassment':[0.09],
        'excitement':[0.22],
        'fear':[0.08],
        'gratitude':[0.28],
        'grief':[0.02],
        'joy':[0.23],
        'love':[0.25],
        'nervousness':[0.07],
        'optimism':[2.63],
        'pride':[0.31],
        'realization':[2.52],
        'relief':[0.17],
        'remorse':[0.05],
        'sadness':[0.14],
        'surprise':[0.14],
        'neutral':[58.83],
        'Followers': [127500],
        'Platform_mapped': [1],
        'AgeRange': [25],
        'ProductCategory_numeric': [1] 
 })

    predictions =model.predict(new_data)
    print("Predicted Values:", predictions)
 

if __name__=='__main__':
    main()

