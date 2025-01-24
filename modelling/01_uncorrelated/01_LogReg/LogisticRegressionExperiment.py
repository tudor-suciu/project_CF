import pandas as pd
import numpy as np
import sys
import time

sys.path.append('../../../')
from utils import modelling as mod
import wandb
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate

# Load the config.yaml file
with open("LogReg.yaml", "r") as file:
    sweep_config = yaml.safe_load(file)

df = pd.read_csv('/Users/tudor/Documents/phd/coding/project_CF/data/final_df_aberdeen.csv')

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

    logreg = LogisticRegression(class_weight = weights,
                                penalty='l1',
                                solver='saga',
                                max_iter=10000,
                                fit_intercept=True,
                                random_state=42)

    cv = KFold(n_splits = 4, shuffle=False)

    start = time.time()
    # Train Logistic Regression with current C
    logreg.C = config.C
    cv_scores = cross_validate(logreg, X_train, y_train, cv=cv, 
                            scoring=('balanced_accuracy', 'jaccard', 'roc_auc_ovr', 'neg_log_loss'),
                            n_jobs=4,
    )   
    print(cv_scores.keys())
    interm = time.time()

    BA_score = cv_scores['test_balanced_accuracy'].mean()
    JAC_score = cv_scores['test_jaccard'].mean()
    AUC_score = cv_scores['test_roc_auc_ovr'].mean()
    CE_score = cv_scores['test_neg_log_loss'].mean()

    logreg.fit(X_train, y_train)

    # Evaluate on the validation fold
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)
    stop = time.time()

    cm = wandb.sklearn.plot_confusion_matrix(y_test, y_pred)
    roc = wandb.sklearn.plot_roc(y_test, y_prob, logreg.classes_)
    sm = wandb.sklearn.plot_summary_metrics(
            logreg, X_train, y_train, X_test, y_test)

    # Log the results
    wandb.log({
        'balanced_accuracy': BA_score,
        'jaccard': JAC_score,
        'roc_auc_ovr': AUC_score,
        'neg_log_loss': CE_score,
        'confusion_matrix': cm,
        'roc': roc,
        'summary_metrics': sm,
        'time': stop - start,
        'time_cv': interm - start,
        })
    
# sweep_id = wandb.sweep(sweep_config, project="logreg-hyperparameter-tuning")
# wandb.agent(sweep_id = 'kpo15hg1', function=train, count=4, project="logreg-hyperparameter-tuning")

if __name__ == '__main__':
    train()