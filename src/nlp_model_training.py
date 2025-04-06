#incomplete code

import argparse
import os
from get_data import get_data, read_params
#NLP model libraries 
import pandas as pd
import spacy
import random
from spacy.tokens import DocBin
from spacy.training.example import Example
from sklearn.metrics import precision_recall_fscore_support
import ast
#multiple regression model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle


def m_regression(config):
        train_df = pd.read_csv(config['processed_data']['train'])  # Load training data
        test_df = pd.read_csv(config['processed_data']['test'])  # Load testing data


        x_train=train_df[['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral','Followers','Platform_mapped','AgeRange','ProductCategory_numeric']]
        y_train=train_df[['SuccessRate','trend_score']]
        
        x_test=test_df[['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral','Followers','Platform_mapped','AgeRange','ProductCategory_numeric']]
        y_test=test_df[['SuccessRate','trend_score']]

        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R² Score:", r2_score(y_test, y_pred))

        with open(config['models']['multiple_regression'], "wb") as file:
            pickle.dump(model, file)
        print("Model saved as regression_model.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../params.yaml')
    args = parser.parse_args()
    config_file=args.config
    config = get_data(config_file)

    if config['trainable']['multiple_regression_train'] == True:
        m_regression(config)
    

if __name__=='__main__':
    main()
