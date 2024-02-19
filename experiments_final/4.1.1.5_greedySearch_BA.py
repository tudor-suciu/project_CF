import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import modelling as mod
from utils import utils
from utils import paths

from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

start_time = time.time()
# -------------------------------------------------

### Prepare the data for looping through it:
print('Preparing the data...')
df = pd.read_csv('/Users/tudor/Documents/phd/coding/project_CF/data/final_df_aberdeen.csv')
df_4 = mod.FloodChoice(df, times1 = False, times4=True)
df_4.drop(['Unnamed: 0','time','time_ok','Unnamed: 0.1'],axis = 1, inplace = True)
x_baseline_train, x_baseline_test, y_baseline_train, y_baseline_test = mod.CustomDataSplit_Nonshuflling(df_4, 'MM', RATIO = 0.8)
weights = mod.CalcClassWeights(df_4['flood'])

df_4 = utils.Add_SquareCube_cols(df_4)
phys_dict = utils.Prepare_PhysicalVar_Dfs(df_4)
# -------------------------------------------------
print('Starting the loop...')
### Experiment LogReg, GridSearch and parameters:

logreg = LogisticRegression(class_weight = weights,
                            penalty='l1',
                            solver='liblinear',
                            max_iter=1000,
                            fit_intercept=True,
                            tol=1e-6,
                            multi_class='ovr')

# Create the KFold cross-validator
cv = KFold(n_splits = 4, shuffle=False)

# Create the grid
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
param_grid = {'C': [1, 1e1, 1e2, 1e3]}
# Create the grid search object
grid_search = GridSearchCV(estimator=logreg, 
                    param_grid=param_grid, 
                    scoring = 'balanced_accuracy',
                    # scoring = 'jaccard',
                    # scoring = 'roc_auc',
                    # scoring = mod.customMetric_estimator_adj,
                    cv=cv, 
                    verbose=1, 
                    n_jobs=6,
)
# -------------------------------------------------

choice_1d_dict, phys_choice_dict, stat_dict = utils.GetChoiceDicts(5100, 50000, 5000000, 500000, 5000000, 50000)
df_of_int = utils.Make_PhyVars_Stats_Choices(df_4,phys_dict, choice_1d_dict, phys_choice_dict, stat_dict)

choices = pd.read_csv('4.1.3.choice_str.csv')


log_file = "4.1.1.5_log.txt"
best_metric = 0

with open(log_file, "w") as file:
    
    file.write('iteration, phys_vars, stats, best_params, tn, fp, fn, tp, BA, AUC, custom \n')
    k=0
    while k < 46:
        k+=1
        best_metric = 0
        best_C = 0
        for i in tqdm(range(len(choices))):
            N = len(choices)
            choice_str_1 = choices.iloc[i]['choice_str_1']
            choice_str_2 = choices.iloc[i]['choice_str_2']
            choice_str_3 = choices.iloc[i]['choice_str_3']
            choice_str_4 = choices.iloc[i]['choice_str_4']
            choice_str_5 = choices.iloc[i]['choice_str_5']
            choice_str_6 = choices.iloc[i]['choice_str_6']

            choice_1d_dict, phys_choice_dict, stat_dict = utils.GetChoiceDicts(choice_str_1, choice_str_2, choice_str_3, choice_str_4, choice_str_5, choice_str_6)
            df_new = utils.Make_PhyVars_Stats_Choices(df_4,phys_dict, choice_1d_dict, phys_choice_dict, stat_dict)

            str1, str2 = utils.GetString_PhysAndStats(choice_str_1, choice_str_2, choice_str_3, choice_str_4, choice_str_5, choice_str_6)
            toWrite_str = str1 +', '+ str2
            file.write(str(i)+'/'+str(N)+','+toWrite_str+', ')

            df_to_test = df_of_int.copy()
            df_to_test[df_new.columns[0]] = df_new[df_new.columns[0]]

            X = df_to_test.drop(['flood'], axis=1)
            y = df_to_test['flood']
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            grid_search.fit(X_scaled, y)
            
            best_params = grid_search.best_params_
            for param, value in best_params.items():
                file.write(f"{param}: {value}"+',  ')

            logreg = LogisticRegression(class_weight = weights,
                                penalty='l1',
                                solver='liblinear',
                                max_iter=1000,
                                C = best_params['C'],
                                fit_intercept=True,
                                tol=1e-6,
                                multi_class='ovr')

            X_train = X_scaled[:int(len(X_scaled)*.8),:]
            y_train = y.iloc[:int(len(X_scaled)*.8):]
            X_test = X_scaled[int(len(X_scaled)*.8):,:]
            y_test = y.iloc[int(len(X_scaled)*.8):]
            y_pred = logreg.fit(X_train, y_train).predict(X_test)

            if metrics.balanced_accuracy_score(y_test, y_pred) > best_metric:
                best_metric = metrics.balanced_accuracy_score(y_test, y_pred)
                best_choice = (choice_str_1, choice_str_2, choice_str_3, choice_str_4, choice_str_5, choice_str_6)
                best_index = i
                best_C = best_params['C']

            # file.write("Classification Metrics:, ")
            file.write(str(confusion_matrix(y_test, y_pred)[0,0])+", ")
            file.write(str(confusion_matrix(y_test, y_pred)[0,1])+", ")
            file.write(str(confusion_matrix(y_test, y_pred)[1,0])+", ")
            file.write(str(confusion_matrix(y_test, y_pred)[1,1])+", ")
            file.write(str(metrics.balanced_accuracy_score(y_test, y_pred))+", ")
            file.write(str(metrics.roc_auc_score(y_test, y_pred))+", ")
            file.write(str(mod.customMetric_adj(y_test, y_pred))+" ")
            file.write("\n")

        # find which param is the best:
        if '1' in str(best_choice[1]):
            pos1 = str(best_choice[1]).find('1')
            in_which1 = 2
        elif '1' in str(best_choice[2]):
            pos1 = str(best_choice[2]).find('1')
            in_which1 = 3
        elif '1' in str(best_choice[3]):
            pos1 = str(best_choice[3]).find('1')
            in_which1 = 4

        # find which stat is the best:
        if '1' in str(best_choice[4]):
            pos2 = str(best_choice[4]).find('1')
            in_which2 = 5
        elif '1' in str(best_choice[5]):
            pos2 = str(best_choice[5]).find('1')
            in_which2 = 6

        # # update the column:
        # new_col = choices[f'choice_str_{in_which}'].copy()
        
        # if in_which == 2:
        #     new_col = new_col + 10**(len(str(choice_str_2))-pos-1)
        #     for j in range(len(new_col)):
        #         x = new_col.iloc[j]
        #         if '2' in str(x):
        #             pos2 = str(x).find('2')
        #             new_col.iloc[j] = x - 10**(len(str(choice_str_2))-pos2-1) 
        
        # elif in_which == 3:
        #     new_col = new_col + 10**(len(str(choice_str_3))-pos-1)
        #     for j in range(len(new_col)):
        #         x = new_col.iloc[j]
        #         if '2' in str(x):
        #             pos2 = str(x).find('2')
        #             new_col.iloc[j] = x - 10**(len(str(choice_str_3))-pos2-1) 
        
        # elif in_which == 4:
        #     new_col = new_col + 10**(len(str(choice_str_4))-pos-1)
        #     for j in range(len(new_col)):
        #         x = new_col.iloc[j]
        #         if '2' in str(x):
        #             pos2 = str(x).find('2')
        #             new_col.iloc[j] = x - 10**(len(str(choice_str_4))-pos2-1) 
        
        # choices[f'choice_str_{in_which}'] = new_col
            
        # update the df_of_int with the best choice:
        choice_1d_dict, phys_choice_dict, stat_dict = utils.GetChoiceDicts(best_choice[0], best_choice[1], best_choice[2], best_choice[3], best_choice[4], best_choice[5])
        df_new = utils.Make_PhyVars_Stats_Choices(df_4,phys_dict, choice_1d_dict, phys_choice_dict, stat_dict)
        df_of_int[df_new.columns[0]] = df_new[df_new.columns[0]]
        choices.drop(best_index, inplace=True)
        choices.reset_index(drop=True, inplace=True)

        file.write('next par, next par, next par, next par, next par, next par, next par, next par, next par, next par, next par')
        file.write(str(df_of_int.columns))
        file.write('best C: ' + str(best_C))

        file.write("\n")


file.close()

print(df_of_int.columns)
print("time elapsed: {:.2f}s".format(time.time() - start_time))