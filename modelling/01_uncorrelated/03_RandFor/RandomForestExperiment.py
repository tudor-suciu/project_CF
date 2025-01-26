import pandas as pd
import numpy as np
import sys
import time

sys.path.append('../../../')
from utils import modelling as mod
import wandb
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import balanced_accuracy_score

# Load the config.yaml file
with open("RandFor.yaml", "r") as file:
    sweep_config = yaml.safe_load(file)

# df = pd.read_csv('/Users/tudor/Documents/phd/coding/project_CF/data/final_df_aberdeen.csv')
df = pd.read_csv('/gws/nopw/j04/ai4er/users/ts809/era5_final/final_df_aberdeen.csv')    # JASMIN

def train():
    weights = mod.CalcClassWeights(df['floods'])
    df_to_analyse = df.drop(columns=['Unnamed: 0.1', 'time', 'Unnamed: 0', 'floods', 'time_ok', 't'])

    X = df_to_analyse.drop(columns=['floods_x4'])
    y = df_to_analyse['floods_x4']

    X_train = X.iloc[0:int(len(X) *.8)] 
    X_test = X.iloc[int(len(X) *.8):int(len(X) *.9)]
    X_val = X.iloc[int(len(X) *.9):]

    y_train = y.iloc[0:int(len(y) *.8)]
    y_test = y.iloc[int(len(y) *.8):int(len(y) *.9)]
    y_val = y.iloc[int(len(y) *.9):]
    
    wandb.init()
    config = wandb.config
    
    if config.scaler == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    randfor = RandomForestClassifier(
        criterion='gini',
        max_features='log2',
        random_state=42,
        class_weight=weights,
        verbose =1,
        n_jobs=4,
    )
    
    if config.max_depth != 'None':
        randfor.max_depth = config.max_depth
    randfor.min_samples_split = config.min_samples_split
    randfor.min_samples_leaf = config.min_samples_leaf
    randfor.n_estimators = config.n_estimators
    randfor.bootstrap = config.bootstrap

    cv = KFold(n_splits = 4, shuffle=False)

    start = time.time()

    cv_scores = cross_validate(randfor, X_train, y_train, cv=cv, 
                            scoring=('balanced_accuracy', 'jaccard', 'roc_auc_ovr', 'neg_log_loss'),
                            n_jobs=4,
                            error_score='raise',
    )   
    print(cv_scores.keys())
    interm = time.time()

    BA_score = cv_scores['test_balanced_accuracy'].mean()
    JAC_score = cv_scores['test_jaccard'].mean()
    AUC_score = cv_scores['test_roc_auc_ovr'].mean()
    CE_score = cv_scores['test_neg_log_loss'].mean()

    randfor.fit(X_train, y_train)

    # Evaluate on the validation fold
    y_pred = randfor.predict(X_test)
    y_prob = randfor.predict_proba(X_test)
    stop = time.time()

    BA_test = balanced_accuracy_score(y_test, y_pred)

    # Log the results
    wandb.log({
        'balanced_accuracy': BA_score,
        'jaccard': JAC_score,
        'roc_auc_ovr': AUC_score,
        'neg_log_loss': CE_score,
        'time': stop - start,
        'time_cv': interm - start,
        'balanced_acc_test': BA_test
        })    

if __name__ == '__main__':
    train()