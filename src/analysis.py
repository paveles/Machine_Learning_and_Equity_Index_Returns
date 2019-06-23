#%% #--------------------------------------------------
""" 
Predicting Equity Premium using Machine Learning Methods - Main File
Project Files
-------------
model_configs.py - configurations of the models
transform_cv.py - new transformation and cross-validation methods used in the project
walkforward_functions.py - new methods that are used to do moving window estimations and analysis 
"""

#%% #--------------------------------------------------
#* Import Main Modules
import warnings
import math
import time
import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
import sys # To caclulate memory usage
import os

#* For Saving Models
import pickle

#* Load Walk-Forward Estimation Functions
from src.walkforward_functions import calculate_r2_wf, calculate_msfe_adjusted, estimate_walk_forward
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score

# #* Create Folders
dir = os.getcwd()
# os.chdir(dir)
# os.makedirs(dir + '/temp', exist_ok = True)
# os.makedirs(dir + '/out/temp', exist_ok = True)
# os.makedirs(dir + '/in', exist_ok = True)
#%% #--------------------------------------------------
#* Load Global Parameters *
#* Load Configs of Different Models
from src.settings import Period, ROLLING, start_idx, Models_Folder,\
VERBOSE, CONFIGS, training_window, validation_window

#%% #--------------------------------------------------
#* Load Data
df = pd.read_pickle("data/processed/df.pickle")
#%% #--------------------------------------------------
#* Define variables
other = ['ewsi']
state = ['recessionD', 'sent']
macro = [ 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] 
tech = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
       'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
       'vol_3_12']
# predictors = macro+ tech # + other  + state
#%% #--------------------------------------------------
#* Variable Cut
if Period == 1974:

    predictors = macro + tech + other  + state

elif Period == 1928:

    predictors = macro
elif Period == 1951:
  
    predictors = macro + tech
else:
    sys.exit("Wrong Sample")

df=df[['date','lnsp500_rf']+predictors]


#%% #--------------------------------------------------
#*"""Lagging predictive  variables"""

df[predictors] = df[predictors].shift(1)

#%% #--------------------------------------------------
#* Sample Cut

if Period == 1974:
    df = df[(df['date'].dt.year >= 1974)&(df['date'].dt.year <= 2010)]


elif Period == 1928:
    df = df[(df['date'].dt.year >= 1928)]

elif Period == 1951:
    df = df[(df['date'].dt.year >= 1951)]

else:
    sys.exit("Wrong Sample")
#df = df.drop(['date'],axis = 1)
# df[pd.isnull(df["ewsi"])!= 1]['date'].describe()
df.dropna(inplace = True)

#%% #--------------------------------------------------
#*"""Provide a Description of the Data"""
df[['lnsp500_rf']+predictors].describe().T.to_csv("out/temp/descriptive.csv")
#""" --> Data is the same is in the paper Rapach et al 2013"""
# df.describe().T
#%% #--------------------------------------------------

#%% #--------------------------------------------------
#""" Define X and Y"""
Xo= df.drop(['lnsp500_rf','date'],axis = 1)
yo = df['lnsp500_rf']

#%% #--------------------------------------------------
# #############################################################################

# #%% #--------------------------------------------------
# #! Do All Time-Consuming Calculations!
# #* Estimating Walk-Forward and Saving Estimation Results
# # Model configurations to be used for estimation - see "model_configs.py" 


# os.makedirs(dir + '/out/'+ Models_Folder +'/pickle', exist_ok = True)
# os.makedirs(dir + '/out/'+ Models_Folder +'/models/estimated', exist_ok = True)

# for cname, config in CONFIGS.items():
#     print('--------------------------')
#     time_begin = datetime.datetime.now()
#     #* Estimate Walk-Forward
#     print(cname +' '+ time_begin.strftime('%Y-%m-%d %H:%M:%S'))
#     estimated = estimate_walk_forward(config ,Xo,yo,start_idx, rolling = ROLLING,
#     tr_win = training_window, val_win = validation_window, verbose = VERBOSE) #! The code

#     time_end = datetime.datetime.now()
#     print(cname +' '+ time_end.strftime('%Y-%m-%d %H:%M:%S'))
#     models_estimated = estimated[0]
#     scores_estimated = estimated[1]
#     y_pred = estimated[2]

#     #%% #--------------------------------------------------
#     #* Save Pickle of the Model and Config
#     config_model_pickle = {'name': config['name'], 'estimated': estimated, 'config': config}
#     with open("out/"+ Models_Folder +"/pickle/"+config['name']+".pickle","wb") as f:
#         pickle.dump(config_model_pickle, f, -1)


#     #%% #--------------------------------------------------
#     #* Calculate different metrics
#     y_true = yo.loc[y_pred.index]
#     #** Calculating Moving Wndow Mean
#     y_moving_mean = yo.shift(1).expanding(1).mean().iloc[start_idx:]
    
#     r2_oos = calculate_r2_wf(y_true, y_pred,y_moving_mean)
#     msfe_adj, p_value = calculate_msfe_adjusted(y_true, y_pred, y_moving_mean)
#     mse_oos = mean_squared_error(y_true,y_pred)
#     mse_validated = - scores_estimated.mean()

#     # print("r2_oos = " + str(r2_oos))
#     # print("(msfe_adj,p_value) = " + str(msfe_adj) + ", "+ str(p_value))
#     # print("mse_oos = " + str(mse_oos))
#     # print("average mse_validated  = " + str(mse_validated))
 
#     #%% #--------------------------------------------------
#     #* Save results_dict to the CSV file
#     results_dict = {}
#     results_dict['name'] = config['name'] 
#     results_dict['r2_oos'] = r2_oos
#     results_dict['msfe_adj'] = msfe_adj
#     results_dict['mse_oos'] = mse_oos
#     results_dict['mse_validated'] = mse_validated
#     results_dict['time_begin'] = time_begin.strftime('%Y-%m-%d %H:%M:%S')
#     results_dict['time_end'] = time_end.strftime('%Y-%m-%d %H:%M:%S')
#     results_dict['time_diff'] = (time_end - time_begin)
#     results_dict['start_idx'] = start_idx
#     results_dict['window_training'] = training_window
#     results_dict['window_validation'] = validation_window
#     results_dict['window'] = Models_Folder
#     results_dict['config'] = str(config)
#     results_dict['period'] = int(Period)

#     df = pd.DataFrame(results_dict, index=[0]) 
#     df.to_csv('out/'+ Models_Folder +'/models/'+ results_dict['name']+'.csv', index=False)
    
#     #* Save Predictions and Scores to a Separate File 
#     model_results = pd.DataFrame()
#     model_results['y_pred'] = y_pred
#     model_results['index'] = y_pred.index 
#     model_results['scores_estimated'] = scores_estimated
#     model_results.to_csv('out/'+ Models_Folder +'/models/'+ results_dict['name']+'_predictions.csv', index=False)


# #%% #--------------------------------------------------
# #* Estimated Models Save in Temp

# for cname, config in CONFIGS.items():
#     with open("out/"+ Models_Folder +"/pickle/" + config['name']+".pickle", "rb") as f:
#         config_model_pickle = pickle.load(f)
#         config_model_pickle['estimated'][0].apply(lambda x: x.named_steps).to_csv(
#             'out/'+ Models_Folder +'/models/estimated/'+ config['name'] +'_estimated.csv',
#              header = True)
# # Lambda Function is used because otherwise not all steps are revealed


#%% #--------------------------------------------------
#* Aggregate Information into one file


df_config = pd.DataFrame()
for cname, config in CONFIGS.items():
    df_config = df_config.append(pd.read_csv('out/'+ Models_Folder +'/models/'+ cname +'.csv'),
     ignore_index =True)
print(df_config)
df_config.to_csv('out/'+ Models_Folder +'/models/'+'All_Models'+'.csv')
#%% #--------------------------------------------------
