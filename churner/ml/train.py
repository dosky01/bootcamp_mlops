# Importer les librairies
import os
import optuna
import yaml
import pickle
import pandas as pd
import numpy as np
import argparse
from functools import partial
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from utils import (loader, 
                      splitter, 
                      preprocessor, 
                      objective)


# use the parser to take the file configuration
parser = argparse.ArgumentParser(description='take the config path.')
parser.add_argument('config_path', metavar='N', type=str)
args = parser.parse_args()
config_path = args.config_path

def training_and_testing_pipeline():
    # load the config file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    # load the dataframe
    df = loader(config['data']['datapath'])

    # Separate the dataset into train/test/calibration sets
    df_train, df_test = splitter(df,
                             config['data']['categorical'],
                             config['data']['numerical'], 
                             config['data']['label'],
                             seed=config.get('seed'),
                             percentage=config['model']['test-set-percentage'])

    df_test, df_cal = splitter(df_test,
                               config['data']['categorical'],
                               config['data']['numerical'], 
                               config['data']['label'],
                               seed=config.get('seed'),
                               percentage=config['model']['calibration-set-percentage'])
    # Preprocessing
    X, y = preprocessor(df_train, config, option_train='train', option_output='all')
    X_test, y_test = preprocessor(df_test, 
                                  config, 
                                  option_train='no-train', 
                                  option_output='all')
    
    X_cal, y_cal = preprocessor(df_cal, 
                                config, 
                                option_train='no-train', 
                                option_output='all')
    
    # Looking for best Hyper-parameters 
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, config=config, X=X, y=y), n_trials=config['model']['n_trials'])
    # Training the best model
    # Instanciation a model with the best hyperparameters
    best_model = XGBClassifier(random_state=config['seed'],
                               use_label_encoder=False,
                               eval_metric='auc',
                               **study.best_params)
    # Training on all the training set
    best_model.fit(X, y)
    # Calibrate the best model with an Isotonic regression
    best_model_calibrated = CalibratedClassifierCV(best_model, 
                                                   cv="prefit", 
                                                   method="isotonic")
    best_model_calibrated.fit(X_cal, y_cal)

    # Save the calibrated model
    model_name = config['model']['name']
    savepath = config['model']['savepath']
    savepath = os.path.join(savepath, model_name)

    with open(savepath, "wb") as file:
        pickle.dump(best_model_calibrated, file)
    print('The whole pipeline is over !')
    


if __name__ == '__main__':
    if os.path.exists(config_path):
        training_and_testing_pipeline()
    else:
        print("file does not exist !!!")