from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ConfusionMatrix(y_tested,y_predicted):
    '''
    Creates the Confusion Matrix according to the custom Visuals.
    ---
    y_tested (): Original data
    y_predicted (): Predicted data
    '''

    cnf_matrix = confusion_matrix(y_tested, y_predicted)

    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True')
    plt.xlabel('Predicted')

#--------------------------------------------------------------------------------------------------


def CustomDataSplit_Nonshuflling(df, scaler_choice, RATIO = 0.8):
    '''
    df
    scaler_choice (str): MM or SS
    RATIO 
    '''
    y = df['flood']
    X = df.drop(['flood'], axis=1)

    # split the data into train and test sets
    x_train = X[:int(len(X)*RATIO)]
    x_test = X[int(len(X)*RATIO):]

    y_train = y[:int(len(y)*RATIO)]
    y_test = y[int(len(y)*RATIO):]

    if scaler_choice == 'MM':
        scaler = MinMaxScaler()
    elif scaler_choice == 'SS':
        scaler = StandardScaler()
    else:
        print('Scaler not defined -- pick MM or SS')

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

#--------------------------------------------------------------------------------------------------

def CalcClassWeights(y_data):
    '''
    Calculate Class Weights for imbalanced data.
    You can select whether you want to use the whole dataset or only the train set,
    by ajdusting the input y_data.
    ---
    y_data (pd.Series): the target variable
    '''
    
    R = y_data.value_counts()[1]/len(y_data)
    weights = {0: R, 1: 1-R}

    return weights

#--------------------------------------------------------------------------------------------------
def customMetric(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))
    
    recall = tp / (tp + fn)
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
    specificity = tn / (tn + fp)
    FPR = fp / (fp + tn)
    custom = recall * balanced_accuracy * (specificity - FPR)
    return custom

#--------------------------------------------------------------------------------------------------
def customMetric_estimator(estimator, X, y_true):
    y_pred = estimator.predict(X)
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))
    
    recall = tp / (tp + fn)
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
    specificity = tn / (tn + fp)
    FPR = fp / (fp + tn)
    custom = recall * balanced_accuracy * (specificity - FPR)
    return custom
#--------------------------------------------------------------------------------------------------

def PrintModelOutput(y_test, y_pred):

    ConfusionMatrix(y_test, y_pred)
    
    print("  Model metrics  : ----------------------")
    print("BALANCED ACCURACY: ", metrics.balanced_accuracy_score(y_test, y_pred))
    print("         Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("        Precision: ", metrics.precision_score(y_test, y_pred))
    print("           Recall: ", metrics.recall_score(y_test, y_pred))
    print("         F1 Score: ", metrics.f1_score(y_test, y_pred))
    print(" my Custom Metric: ", customMetric(y_test, y_pred))
    print("------------------------------------------")

#--------------------------------------------------------------------------------------------------

def PrepareData():
    df = pd.read_csv('ABR_ERA5_with_tides.csv')
    df_nao = pd.read_csv('../data/abr_era5_1979_2018_4xfloods.csv')

    # get only the NAO data from the df_nao, del df_nao
    df_only_nao = df_nao[['t', 'nao']]
    del df_nao
    # round the t column to 5 decimals, to be able to merge the 2 dfs
    df['t'] = round(df['t'],5)
    df_only_nao['t'] = round(df_only_nao['t'],5)
    df = df.merge(df_only_nao, on='t', how='left')
    # drop extra columns
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.drop(['time64'],axis=1,inplace=True)

    return df

#--------------------------------------------------------------------------------------------------

def FloodChoice(final_df,times1 = False, times4=False):
    if times1 & times4:
        print('You cannot choose both - please make sure you only choose one value to be true.')
    else:
        if times1 == True:
            final_df.drop('floods_x4',axis = 1, inplace = True)
            final_df.rename(columns={'floods': 'flood'}, inplace=True)
        elif times4 == True:
            final_df.drop('floods',axis = 1, inplace = True)
            final_df.rename(columns={'floods_x4': 'flood'}, inplace=True)
        else:
            print('Please choose which column to use as the flood column - times1 or times4')
    return final_df


