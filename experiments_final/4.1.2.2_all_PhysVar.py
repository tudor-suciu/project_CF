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

out_file = '4.1.2.2_log.txt'
sys.stdout = open(out_file, 'w')

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
                            max_iter=10000,
                            fit_intercept=True,
                            tol=1e-6,
                            multi_class='ovr')

# Create the KFold cross-validator
cv = KFold(n_splits = 4, shuffle=False)

# Create the grid
# param_grid = {'C': [1, 1e1, 1e2, 1e3]}
# param_grid = {'C': [1, 1e1]}
param_grid = {'C': [1e2]}

# Create the grid search object
grid_search = GridSearchCV(estimator=logreg, 
                    param_grid=param_grid, 
                    # scoring = mod.customMetric_estimator_adj,
                    scoring = 'jaccard',
                    cv=cv, 
                    verbose=2, 
                    n_jobs=8,
)
# -------------------------------------------------

print('All PhysVars, 4.1.2.2\n')
choice_str_1 = 5111
choice_str_2 = 51010
choice_str_3 = 5111100
choice_str_4 = 511111
choice_str_5 = 5111100
choice_str_6 = 50000

choice_1d_dict, phys_choice_dict, stat_dict = utils.GetChoiceDicts(choice_str_1, choice_str_2, choice_str_3, choice_str_4, choice_str_5, choice_str_6)
df_new = utils.Make_PhyVars_Stats_Choices(df_4,phys_dict, choice_1d_dict, phys_choice_dict, stat_dict)

X = df_new.drop(['flood'], axis=1)
y = df_new['flood']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

grid_search.fit(X_scaled, y)
best_params = grid_search.best_params_

logreg = LogisticRegression(class_weight = weights,
            penalty='l1',
            solver='liblinear',
            max_iter=100,
            C = best_params['C'],
            fit_intercept=True,
            tol=1e-6,
            multi_class='ovr')

X_train = X_scaled[:int(len(X_scaled)*.8),:]
y_train = y.iloc[:int(len(X_scaled)*.8):]
X_test = X_scaled[int(len(X_scaled)*.8):,:]
y_test = y.iloc[int(len(X_scaled)*.8):]
        
print("Best parameters found:")
print(grid_search.best_params_) 
print("\n")

sys.stdout.close()
y_pred = logreg.fit(X_train, y_train).predict(X_test)

out_file = '4.1.2.2_log2.txt'
sys.stdout = open(out_file, 'w')

print("Classification report:")
# file.write("Classification Metrics:, ")
print(str(confusion_matrix(y_test, y_pred)[0,0])+", ")
print(str(confusion_matrix(y_test, y_pred)[0,1])+", ")
print(str(confusion_matrix(y_test, y_pred)[1,0])+", ")
print(str(confusion_matrix(y_test, y_pred)[1,1])+", ")
print(str(metrics.balanced_accuracy_score(y_test, y_pred))+", ")
print(str(metrics.roc_auc_score(y_test, y_pred))+", ")
print(str(mod.customMetric_adj(y_test, y_pred))+" ")
print("\n")

coeffs = pd.DataFrame({'coeff': logreg.coef_[0]}, 
             index=X.columns)
coeffs['exp_coeff'] = np.exp(coeffs['coeff'])*100
coeffs = coeffs.sort_values(by = 'exp_coeff', ascending = False)

print(coeffs)

print("time elapsed: {:.2f}s".format(time.time() - start_time))
sys.stdout.close()
# print("time elapsed: {:.2f}s".format(time.time() - start_time))