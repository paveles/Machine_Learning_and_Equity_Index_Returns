
#%% [markdown] #--------------------------------------------------
## Equity Premium and Machine Learning
#%% #--------------------------------------------------
#* Import Modules
import warnings
import math
import time
import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
import sys # To caclulate memory usage
import os

dir = os.getcwd()
os.chdir(dir)
os.makedirs(dir + '/temp', exist_ok = True)
os.makedirs(dir + '/out/temp', exist_ok = True)
os.makedirs(dir + '/out/pickle', exist_ok = True)
os.makedirs(dir + '/in', exist_ok = True)

#%% #--------------------------------------------------
#* Global Parameters *
# Cross-validation Parameter
K = 10
#  Share of Sample as Test
TsizeInv = 10
test_size= 1/TsizeInv
# Add interactions or not
Poly = 1
# Starting Year
Period  = 1951
# Number of Lags
LAGS = 1

# Estimate using Rolling Window or Exapnding
ROLLING = True
min_idx = 0
start_idx = 240


if ROLLING == True:
    Models_Folder = 'rolling'
else:
    Models_Folder = 'expanding'

#%% #--------------------------------------------------
#* Load Data
df = pd.read_csv('in/rapach_2013.csv', na_values = ['NaN'])
df.rename( index=str, columns={"date": "ym"}, inplace=True)
df['date'] = pd.to_datetime(df['ym'],format='%Y%m') + MonthEnd(1)
df['sp500_rf'] = df['sp500_rf'] * 100
df['lnsp500_rf'] = df['lnsp500_rf'] * 100
df = df.sort_values(by=['date'])
df.index = df.index.astype(int)
df0 = df

#df = df.set_index(['ym'])

#%% #--------------------------------------------------
"""
Define variables
"""
other = ['ewsi']
state = ['recessionD', 'sent']
macro = [ 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] 
tech = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
       'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
       'vol_3_12']
# predictors = macro+ tech # + other  + state
#%% #--------------------------------------------------
"""
Variable Cut
"""

if Period == 1974:

    predictors = macro + tech + other  + state

elif Period == 1928:

    predictors = macro
elif Period == 1951:
  
    predictors = macro + tech
else:
    sys.exit("Wrong Sample")

df=df[['date','lnsp500_rf']+predictors]
# df[pd.isnull(df["ewsi"])!= 1]['date'].describe()
#df = df.set_index(['date'])

#%% #--------------------------------------------------
#*"""Lagging predictive  variables"""

df[predictors] = df[predictors].shift(1)
if LAGS>1:
    for lag in range(2,LAGS+1,1):
        df = pd.concat([df, df[predictors].shift(lag).add_suffix('_l{}'.format(lag))], axis = 1)

#%% #--------------------------------------------------
"""
Sample Cut
"""

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
#''' Define X and Y'''
from sklearn.model_selection import train_test_split
Xo= df.drop(['lnsp500_rf','date'],axis = 1)
yo = df['lnsp500_rf']

#%% #--------------------------------------------------
# #############################################################################

#%% #--------------------------------------------------
#''' Interaction Terms'''
from sklearn.preprocessing import PolynomialFeatures
if Poly == 1:
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    Xp = poly.fit_transform(Xo)

elif Poly == 2:
    poly = PolynomialFeatures(degree = 2,include_bias = False)
    Xp = poly.fit_transform(Xo)

else:
    Xp = Xo



#%% #--------------------------------------------------
#* Walk-Forward Modeling
from sklearn.linear_model import  LinearRegression
from scipy import stats
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from helper import TimeSeriesSplitMod
from helper import DisabledCV, ToConstantTransformer, ToNumpyTransformer

def calculate_r2_wf(y_true, y_pred, y_moving_mean):
    '''
    Calculate out-of-sample R^2 for the walk-forward procedure
    '''
    mse_urestricted = ((y_true - y_pred)**2).sum()
    mse_restricted = ((y_true - y_moving_mean)**2).sum()
    return 1 - mse_urestricted/mse_restricted

def calculate_msfe_adjusted(y_true, y_pred, y_moving_mean):
    f = (y_true - y_moving_mean)**2 - ((y_true - y_pred)**2 - (y_moving_mean - y_pred)**2)
    t_stat,pval_two_sided = stats.ttest_1samp(f, 0, axis=0)
    pval_one_sided = stats.t.sf(t_stat, f.count() - 1)
    return t_stat, pval_one_sided


def estimate_walk_forward(config, X, y, start_idx, max_idx, rolling = False, verbose = True):
    #print(config['name']+' '+ str(time.localtime(time.time())))
    if verbose == True:
        print(config['param_grid'])

    models_estimated = pd.Series(index=X.index[start_idx:])
    scores_estimated = pd.Series(index=X.index[start_idx:])
    predictions = pd.Series(index=X.index[start_idx:])
    model_to_estimate = config['pipeline']

    
    scorer = config['scorer']
    grid_search = config['grid_search']
    param_grid = config['param_grid']



    for idx in range(start_idx,max_idx,1):    
        if rolling == True:
            X_tr = X.iloc[idx - 240 : idx]
            y_tr = y.iloc[idx - 240 : idx]
            if config['cv'] == TimeSeriesSplitMod:
                cv = config['cv']( n_splits =240 - 1, start_test_split = 240-24 ).split(X_tr,y_tr)
            elif config['cv'] == DisabledCV:
                cv = config['cv']().split(X_tr,y_tr)
        else:
            X_tr = X.iloc[0 : idx]
            y_tr = y.iloc[0 : idx]
            if config['cv'] == TimeSeriesSplitMod:
                cv = config['cv']( n_splits =idx - 1, start_test_split = start_idx - 1 ).split(X_tr,y_tr)
            elif config['cv'] == DisabledCV:
                cv = config['cv']().split(X_tr,y_tr)
        
        grid = grid_search(estimator=model_to_estimate, param_grid=param_grid, cv=cv \
            , scoring = scorer, n_jobs=-1)        
        model = grid.fit(X_tr,y_tr)
        best_model = model.best_estimator_
        best_score = model.best_score_
        models_estimated.loc[X.index[idx]] = best_model # save the model
        if verbose == True:        
            if ((idx-start_idx) % 10) == 0:
                print(str(idx)+" / "+str(max_idx) )
                print(best_model)
                print(best_score)

        scores_estimated.loc[X.index[idx]] = best_score # save the score
        predictions.loc[X.index[idx]] = model.predict([X.iloc[idx]]) # predict next month 
    return models_estimated,scores_estimated, predictions
#%% #--------------------------------------------------
# #* Check How Indexes and Cross-Validation are Calculated
# idx = len(yo)-1
# tscv = TimeSeriesSplitMod( n_splits = 180 - 1, start_test_split = 156 ) #( n_splits =idx - 1, start_test_split = start_idx - 1 )

# # print(tscv)  
# X_tr = Xo.iloc[idx - 180 : idx] # Xo[0:idx]
# y_tr = yo.iloc[idx - 180 : idx] # yo[0:idx]
# i = 0
# for train_index, test_index in tscv.split(X_tr, y_tr):
#     i+=1
#     print("TRAIN:", y_tr.index[train_index], "TEST:", y_tr.index[test_index])
#     print( len(y_tr.index[train_index]), len(y_tr.index[test_index]))
# #print(Xo.index[idx])
# print(i)
# print( idx)
# ##* Correct!
   
#%% #--------------------------------------------------
#! Do All Time-Consuming Calculations!
from model_configs import *
configs ={
    # 'const' : const_config,
    'ols' : ols_config,
    # 'pca' : pca_config, #~ 23 minutes
    # 'enet' : enet_config, #~ 2.5 hours
    # 'pca_enet' : pca_enet_config, #~ 3 hours
    # 'adab_nocv' : adab_nocv_config,
    # 'gbr_nocv': gbr_nocv_config,
    # 'rf_nocv': rf_nocv_config,
    # 'xgb_nocv': xgb_nocv_config,
    # 'adab': adab_config,
    # 'gbr': gbr_config,
    # 'rf': rf_config,
    # 'xgb' : xgb_config
}
#config = ols_config

os.makedirs(dir + '/out/'+ Models_Folder +'/pickle', exist_ok = True)
os.makedirs(dir + '/out/'+ Models_Folder +'/models/estimated', exist_ok = True)

for cname, config in configs.items():
    print('--------------------------')
    time_begin = datetime.datetime.now()
    print(cname +' '+ time_begin.strftime('%Y-%m-%d %H:%M:%S'))
    max_idx = yo.shape[0]
    estimated = estimate_walk_forward(config ,Xo,yo,start_idx,max_idx, rolling = ROLLING) #! The code

    time_end = datetime.datetime.now()
    print(cname +' '+ time_end.strftime('%Y-%m-%d %H:%M:%S'))
    models_estimated = estimated[0]
    scores_estimated = estimated[1]
    y_pred = estimated[2]

    #models_estimated, scores_estimated, y_pred
    #%% #--------------------------------------------------
    #* Save Pickle of the Model and Config
    import pickle
    config_model_pickle = {'name': config['name'], 'estimated': estimated, 'config': config}
    with open("out/"+ Models_Folder +"/pickle/"+config['name']+".pickle","wb") as f:
        pickle.dump(config_model_pickle, f, -1)


    #%% #--------------------------------------------------
    #* Calculate different metrics
    from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
    import time
    y_true = yo.loc[y_pred.index]

    y_moving_mean = yo.shift(1).expanding(1).mean().iloc[start_idx:]

    r2_oos = calculate_r2_wf(y_true, y_pred,y_moving_mean)
    print("r2_oos = " + str(r2_oos))
    msfe_adj, p_value = calculate_msfe_adjusted(y_true, y_pred, y_moving_mean)
    print("(msfe_adj,p_value) = " + str(msfe_adj) + ", "+ str(p_value))
    mse_oos = mean_squared_error(y_true,y_pred)
    print("mse_oos = " + str(mse_oos))
    mse_validated = - scores_estimated.mean()
    print("average mse_validated  = " + str(mse_validated))
    #print(models_estimated[-1])

    #ticks = time.time()
    #models_estimated.to_csv('temp\models_estimated'+str(ticks)+'.csv', header = True)
    #%% #--------------------------------------------------
    #* Save results_dict to CSV file
    results_dict = {}
    results_dict['name'] = config['name'] 
    results_dict['r2_oos'] = r2_oos
    results_dict['msfe_adj'] = msfe_adj
    results_dict['mse_oos'] = mse_oos
    results_dict['mse_validated'] = mse_validated
    results_dict['time_begin'] = time_begin.strftime('%Y-%m-%d %H:%M:%S')
    results_dict['time_end'] = time_end.strftime('%Y-%m-%d %H:%M:%S')
    results_dict['time_diff'] = (time_end - time_begin)
    results_dict['start_idx'] = start_idx   
    results_dict['config'] = str(config)
    results_dict['period'] = int(Period)

    df = pd.DataFrame(results_dict, index=[0]) 
    df.to_csv('out/'+ Models_Folder +'/models/'+ results_dict['name']+'.csv', index=False)

    model_results = pd.DataFrame()
    model_results['y_pred'] = y_pred
    model_results['index'] = y_pred.index 
    model_results['scores_estimated'] = scores_estimated
    model_results.to_csv('out/'+ Models_Folder +'/models/'+ results_dict['name']+'_predictions.csv', index=False)



#%% #--------------------------------------------------
#* Aggregate Information
configs ={
    # 'const' : const_config,
    'ols' : ols_config,
    # 'pca' : pca_config,
    # 'enet' : enet_config,
    # 'pca_enet' : pca_enet_config,
    # 'adab_nocv' : adab_nocv_config,
    # 'gbr_nocv': gbr_nocv_config,
    # 'rf_nocv': rf_nocv_config,
    # 'xgb_nocv': xgb_nocv_config,
    # 'adab' : adab_config,
    # 'gbr':gbr_config,
    # 'rf': rf_config,
    # 'xgb': xgb_config,
}

df_config = pd.DataFrame()
for cname, config in configs.items():
    df_config = df_config.append(pd.read_csv('out/'+ Models_Folder +'/models/'+ cname +'.csv'),
     ignore_index =True)
print(df_config)
df_config.to_csv('out/'+ Models_Folder +'/models/'+'All_Models'+'.csv')
#%% #--------------------------------------------------
#* Estimated Models Save in Temp
for cname, config in configs.items():
    with open("out/"+ Models_Folder +"/pickle/" + config['name']+".pickle", "rb") as f:
        config_model_pickle = pickle.load(f)
        config_model_pickle['estimated'][0].apply(lambda x: x.named_steps).to_csv(
            'out/'+ Models_Folder +'/models/estimated/'+ config['name'] +'_estimated.csv',
             header = True)
#* Lambda Function is used because otherwise not all steps are revealed
#%% #--------------------------------------------------
