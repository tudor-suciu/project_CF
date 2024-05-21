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
                            penalty='l2',
                            solver='liblinear',
                            max_iter=1000,
                            fit_intercept=True,
                            tol=1e-6,
                            multi_class='ovr')

# Create the KFold cross-validator
cv = KFold(n_splits = 4, shuffle=False)

# Create the grid
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
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

choices_4_1_1 = pd.read_csv('4.1.2.choice_str.csv')
N = len(choices_4_1_1)

log_file = "outputs/4.1.1.1_log.txt"
with open(log_file, "w") as file:
    file.write('iteration, phys_vars, stats, best_params, tn, fp, fn, tp, BA, AUC, custom,JAC ,Centropy, coeffs, exp_coeffs \n')
    for i in tqdm(range(len(choices_4_1_1))):
    
        choice_str_1 = choices_4_1_1.iloc[i]['choice_str_1']
        choice_str_2 = choices_4_1_1.iloc[i]['choice_str_2']
        choice_str_3 = choices_4_1_1.iloc[i]['choice_str_3']
        choice_str_4 = choices_4_1_1.iloc[i]['choice_str_4']
        choice_str_5 = choices_4_1_1.iloc[i]['choice_str_5']
        choice_str_6 = choices_4_1_1.iloc[i]['choice_str_6']

        choice_1d_dict, phys_choice_dict, stat_dict = utils.GetChoiceDicts(choice_str_1, choice_str_2, choice_str_3, choice_str_4, choice_str_5, choice_str_6)
        df_new = utils.Make_PhyVars_Stats_Choices(df_4,phys_dict, choice_1d_dict, phys_choice_dict, stat_dict)

        str1, str2 = utils.GetString_PhysAndStats(choice_str_1, choice_str_2, choice_str_3, choice_str_4, choice_str_5, choice_str_6)
        toWrite_str = str1 +', '+ str2
        file.write(str(i)+'/'+str(N)+','+toWrite_str+', ')

        X = df_new.drop(['flood'], axis=1)
        y = df_new['flood']
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train = X_scaled[:int(len(X_scaled)*.8),:]
        y_train = y.iloc[:int(len(X_scaled)*.8):]
        X_test = X_scaled[int(len(X_scaled)*.8):,:] 
        y_test = y.iloc[int(len(X_scaled)*.8):]

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        for param, value in best_params.items():
            file.write(f"{param}: {value}"+',  ')

        logreg = LogisticRegression(class_weight = weights,
                            penalty='l2',
                            solver='liblinear',
                            max_iter=5000,
                            C = best_params['C'],
                            fit_intercept=True,
                            tol=1e-6,
                            multi_class='ovr')

        y_pred = logreg.fit(X_train, y_train).predict(X_test)

        # file.write("Classification Metrics:, ")
        file.write(str(confusion_matrix(y_test, y_pred)[0,0])+", ")
        file.write(str(confusion_matrix(y_test, y_pred)[0,1])+", ")
        file.write(str(confusion_matrix(y_test, y_pred)[1,0])+", ")
        file.write(str(confusion_matrix(y_test, y_pred)[1,1])+", ")
        file.write(str(metrics.balanced_accuracy_score(y_test, y_pred))+", ")
        file.write(str(metrics.roc_auc_score(y_test, y_pred))+", ")
        file.write(str(mod.customMetric_adj(y_test, y_pred))+", ")
        file.write(str(metrics.jaccard_score(y_test,y_pred))+", ")
        file.write(str(metrics.log_loss(y_test,y_pred))+", ")
        
        coeffs = pd.DataFrame({'coeff': logreg.coef_[0]}, 
                    index=X.columns)
        coeffs['exp_coeff'] = np.exp(coeffs['coeff'])*100
        coeffs = coeffs.sort_values(by = 'exp_coeff', ascending = False)

        file.write(str(coeffs.coeff.values)+ ", ")
        file.write(str(coeffs.exp_coeff.values)+ " ")

        file.write("\n")
file.close()

print("time elapsed: {:.2f}s".format(time.time() - start_time))