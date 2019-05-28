#%% [markdown] #--------------------------------------------------
## Equity Premium and Machine Learning
#%% #--------------------------------------------------

import warnings
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
import math
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
sns.set()
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
import sys # To caclulate memory usage
import os

dir = os.getcwd()
os.chdir(dir)
os.makedirs(dir + '/temp', exist_ok = True)
os.makedirs(dir + '/out/temp', exist_ok = True)
os.makedirs(dir + '/out/pickle', exist_ok = True)
os.makedirs(dir + '/in', exist_ok = True)



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
#%% #--------------------------------------------------

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

# df['recessionD_c'] = df['recessionD']
# vars = ['recessionD', 'dp', 'dy', 'ep', 'de', \
#        'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl', \
#        'ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
#        'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
#        'vol_3_12', 'sento ', 'sent', 'dsento', 'dsent', 'ewsi']
# Important! Lagging by 1

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
    # #############################################################################

#%% #--------------------------------------------------
#''' Train and Test Samples'''
from sklearn.model_selection import train_test_split
Xo= df.drop(['lnsp500_rf','date'],axis = 1)
yo = df['lnsp500_rf']

X0, X0_test, y, y_test = train_test_split(Xo, yo, test_size=test_size, shuffle = False )
#%% #--------------------------------------------------
#'''Standardize Data'''
from sklearn.preprocessing import StandardScaler,MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

pipline = Pipeline(steps=[
#    ('pca', PCA()), # (n_components=4)
    ('minmax', StandardScaler()),
])
scaler = pipline.fit(X0)

#scaler = StandardScaler().fit(X0)
X = pd.DataFrame(scaler.transform(X0),  index=X0.index )# , columns=X.columns
X_test = pd.DataFrame(scaler.transform(X0_test),  index=X0_test.index) #, columns=X_test.columns 
#%% #--------------------------------------------------
#''' Interaction Terms'''
from sklearn.preprocessing import PolynomialFeatures
if Poly == 1:
    poly = PolynomialFeatures(interaction_only=True,include_bias = False)
    Xp = poly.fit_transform(X)
    Xp_test = poly.fit_transform(X_test)
elif Poly == 2:
    poly = PolynomialFeatures(degree = 2,include_bias = False)
    Xp = poly.fit_transform(X)
    Xp_test = poly.fit_transform(X_test)
else:
    Xp = X
    Xp_test = X_test

#%% #--------------------------------------------------
#* Ones for Constant Model
Ones = pd.DataFrame(np.ones(y.shape[0]))
Ones_test = pd.DataFrame(np.ones(y_test.shape[0]))
#%% #--------------------------------------------------
#* Prepare data for the PCA
from sklearn import linear_model
from sklearn.decomposition import PCA
#pca = PCA().fit(X)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
scaler = StandardScaler().fit(X0)
Xscaled = scaler.transform(X0)
Xscaled_test = scaler.transform(X0_test)
pca = PCA(n_components=4)
pca.fit(X0)
X_pca = pca.transform(Xscaled)
X_test_pca = pca.transform(Xscaled_test)


X#%% #--------------------------------------------------
#* Walk-Forward Modeling
from sklearn.linear_model import  LinearRegression
from scipy import stats
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from TimeSeriesSplitMod import TimeSeriesSplitMod
from helper import DisabledCV, ToConstantTransformer, ToNumpyTransformer

def r2_adj_score(y_true,y_pred,N,K):
    r2 = r2_score(y_true,y_pred)
    return 1-(1-r2)*(N-1)/(N-K-1)

r2_scorer = make_scorer(r2_score,greater_is_better=True)


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


def estimate_walk_forward(config, X, y, start_idx, max_idx):
    #print(config['name']+' '+ str(time.localtime(time.time())))
    models_estimated = pd.Series(index=X.index[start_idx:])
    scores_estimated = pd.Series(index=X.index[start_idx:])
    predictions = pd.Series(index=X.index[start_idx:])
    #* moving mean of lagged y
    #param_dict ={'ols__fit_intercept':[True,False],}
    for idx in range(start_idx,max_idx,1):
        if ((idx-start_idx) % 1) == 0:
            print(str(idx)+" / "+str(max_idx) )

        X_tr = X.iloc[0 : idx]
        y_tr = y.iloc[0 : idx]
        model_to_estimate = config['pipeline']
        if config['cv'] == TimeSeriesSplitMod:
            cv = config['cv']( n_splits =idx - 1, start_test_split = start_idx - 1 ).split(X_tr,y_tr)
        elif config['cv'] == DisabledCV:
            cv = config['cv']().split(X_tr,y_tr)
        
        # tscv =  config['cv']( n_splits =idx - 1, start_test_split = start_idx - 1 )
        # for train_index, test_index in tscv.split(X_tr,y_tr):
        #     print("TRAIN:", train_index, "TEST:", test_index)


        scorer = config['scorer']
        grid_search = config['grid_search']
        param_grid = config['param_grid']
        grid = grid_search(estimator=model_to_estimate, param_grid=param_grid, cv=cv \
            , scoring = scorer, n_jobs=-1)

        model = grid.fit(X_tr,y_tr)
        best_model = model.best_estimator_
        best_score = model.best_score_
        models_estimated.loc[X.index[idx]] = best_model # save the model
        scores_estimated.loc[X.index[idx]] = best_score # save the score
        predictions.loc[X.index[idx]] = model.predict([X.iloc[idx]]) # predict next month 
    return models_estimated,scores_estimated, predictions
# #%% #--------------------------------------------------
# #* Check How Indexes and Cross-Validation are Calculated
# idx = max_idx
# tscv = TimeSeriesSplitMod(n_splits=idx - 1, start_test_split=start_idx - 1)

# print(tscv)  

# for train_index, test_index in tscv.split(Xo[0:idx], yo[0:idx] ):
#    print("TRAIN:", train_index, "TEST:", test_index)

# #print(Xo.index[idx])
# print( idx)
##* Correct!
   
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
    # 'gbr': gbr_config,
    # 'rf': rf_config,   
    # 'lgb' : config_lgb,

#    'tpot': config_tpot,
}
#config = ols_config

min_idx = 0
start_idx = 720
max_idx = yo.shape[0]

for cname, config in configs.items():
    print('--------------------------')
    time_begin = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(cname +' '+ time_begin)

    estimated = estimate_walk_forward(config ,Xo,yo,start_idx,max_idx) #! The code

    time_end = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(cname +' '+ time_end)
    models_estimated = estimated[0]
    scores_estimated = estimated[1]
    y_pred = estimated[2]

    #models_estimated, scores_estimated, y_pred
    #%% #--------------------------------------------------
    #* Save Pickle of the Model and Config
    import pickle
    config_model_pickle = {'name': config['name'], 'estimated': estimated, 'config': config}
    with open("out/pickle/"+config['name']+".pickle","wb") as f:
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
    results_dict['time_begin'] = time_begin
    results_dict['time_end'] = time_end     
    results_dict['start_idx'] = start_idx   
    results_dict['config'] = str(config)
    results_dict['period'] = int(Period)

    df = pd.DataFrame(results_dict, index=[0]) 
    df.to_csv('out/models/'+ results_dict['name']+'.csv', index=False)

    model_results = pd.DataFrame()
    model_results['y_pred'] = y_pred
    model_results['index'] = y_pred.index 
    model_results['scores_estimated'] = scores_estimated
    model_results.to_csv('out/models/'+ results_dict['name']+'_predictions.csv', index=False)

    # results_dict['scores_estimated'] = scores_estimated.tolist()
    # results_dict['y_pred'] = y_pred.tolist()
    # results_dict['index'] = y_pred.index.tolist()  

#%% #--------------------------------------------------
#* Aggregate Information
configs ={
    'const' : const_config,
    'ols' : ols_config,
    'pca' : pca_config,
    'enet' : enet_config,
    'pca_enet' : pca_enet_config,
    'adab_nocv' : adab_nocv_config,
    'gbr_nocv': gbr_nocv_config,
 #   'rf_nocv': rf_nocv_config,
    'xgb_nocv': xgb_nocv_config,
    # 'adab' : adab_config,
    'gbr':gbr_config,
    'rf': rf_config,
    # 'xgb': xgb_config,
    # 'lgb' : config_lgb,
#    'tpot': config_tpot,
}

df_config = pd.DataFrame()
for cname, config in configs.items():
    df_config = df_config.append(pd.read_csv('out/models/'+ cname +'.csv'),
     ignore_index =True)
print(df_config)
df_config.to_csv('out/pickle/'+'All_Models'+'.csv')
#%% #--------------------------------------------------

#* Estimated Models Save in Temp
for cname, config in configs.items():
    print(config['name'])
    with open("out/pickle/" + config['name']+".pickle", "rb") as f:
        config_model_pickle = pickle.load(f)
        config_model_pickle['estimated'][0].to_csv('out/temp/'+ config['name'] +'_estimated.csv', header = True)

#%% #--------------------------------------------------


#%%
